import torch
import torch.nn as nn
import torch.nn.functional as F
from .detr_segmentation import expand_multi_length
from .ops.modules import MSDeformAttn
import torchvision.ops


class SingleLevelMaskHeadLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(dim_in, dim_out, 3, padding=1),
            torch.nn.GroupNorm(8, dim_out),
        )

    def forward(self, x):
        return F.relu(self.layer(x))


class InstanceSegmSumBatchMaskHead(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()
        inter_dims = [
            dim,
            context_dim // 2,
            context_dim // 4,
            context_dim // 8,
            context_dim // 16,
            context_dim // 64]

        # H/64 W/64
        self.lvl0_conv1 = SingleLevelMaskHeadLayer(dim, dim)
        self.lvl0_conv2 = SingleLevelMaskHeadLayer(dim, inter_dims[1])

        # H/32 W/32
        self.lvl1_conv = SingleLevelMaskHeadLayer(dim, inter_dims[1])
        self.lvl1_merge = SingleLevelMaskHeadLayer(inter_dims[1], inter_dims[1])

        # H/16 W/16
        self.lvl2_conv = SingleLevelMaskHeadLayer(dim, inter_dims[1])
        self.lvl2_merge = SingleLevelMaskHeadLayer(inter_dims[1], inter_dims[2])

        # H/8 W/8
        self.lvl3_conv = SingleLevelMaskHeadLayer(dim, inter_dims[2])
        self.lvl3_merge = SingleLevelMaskHeadLayer(inter_dims[2], inter_dims[3])

        # H/4 W/4
        self.lvl4_merge = SingleLevelMaskHeadLayer(inter_dims[3], inter_dims[4])
        self.out_lay = torch.nn.Conv2d(inter_dims[4], 1, 3, padding=1)

        self.adapter_fpn = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    # x -> compressed ch backbone feats bbox_masks -> attention maps computed at each level fpn_feat -> Highest spatial res feature from the
    # backbone that in this case is not use by the transformer encoder, so we need to reduce channels
    def forward(self, compressed_backbone_feats, bbox_mask, features, instances_per_batch):

        # H/64 W/64
        x_0 = torch.cat([expand_multi_length(compressed_backbone_feats, instances_per_batch), bbox_mask[0]], 1)
        x_0 = self.lvl0_conv1(x_0)
        x_0 = self.lvl0_conv2(x_0)

        # H/32 W/32
        x_1 = torch.cat([expand_multi_length(features[0], instances_per_batch), bbox_mask[1]], 1)
        x_1 = self.lvl1_conv(x_1)
        x_1 = x_1 + F.interpolate(x_0, size=x_1.shape[-2:], mode="nearest")
        x_1 = self.lvl1_merge(x_1)

        # H/16 W/16
        x_2 = torch.cat([expand_multi_length(features[1], instances_per_batch), bbox_mask[2]], 1)
        x_2 = self.lvl2_conv(x_2)
        x_2 = x_2 + F.interpolate(x_1, size=x_2.shape[-2:], mode="nearest")
        x_2 = self.lvl2_merge(x_2)

        # H/8 W/8
        x_3 = torch.cat([expand_multi_length(features[2], instances_per_batch), bbox_mask[3]], 1)
        x_3 = self.lvl3_conv(x_3)
        x_3 = x_3 + F.interpolate(x_2, size=x_3.shape[-2:], mode="nearest")
        x_3 = self.lvl3_merge(x_3)

        # H/4 W/4
        last_fpn = self.adapter_fpn(features[3])
        last_fpn = expand_multi_length(last_fpn, instances_per_batch)

        x_out = last_fpn + F.interpolate(x_3, size=last_fpn.shape[-2:], mode="nearest")
        x_out = self.lvl4_merge(x_out)
        x_out = self.out_lay(x_out)

        return x_out


class MultiScaleMHAttentionMap(nn.Module):

    def __init__(self, query_dim, hidden_dim, num_heads, num_levels, dropout=0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        for i in range(num_levels):
            setattr(self, f"q_linear_{i}", nn.Linear(query_dim, hidden_dim, bias=bias))
            setattr(self, f"k_linear_{i}", nn.Linear(query_dim, hidden_dim, bias=bias))
            nn.init.zeros_(getattr(self, f"k_linear_{i}").bias)
            nn.init.zeros_(getattr(self, f"q_linear_{i}").bias)
            nn.init.xavier_uniform_(getattr(self, f"k_linear_{i}").weight)
            nn.init.xavier_uniform_(getattr(self, f"q_linear_{i}").weight)

        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def _check_input(self, k, mask):
        assert len(k) == self.num_levels
        if mask is not None:
            assert len(mask) == self.num_levels

    def forward(self, q, k, mask=None):
        self._check_input(k, mask)
        out_multi_scale_maps = []

        for i, k_lvl in enumerate(k):
            q_lvl = q
            q_lvl = getattr(self, f"q_linear_{i}")(q_lvl)

            k_lvl = F.conv2d(k_lvl, getattr(self, f"k_linear_{i}").weight.unsqueeze(-1).unsqueeze(-1), getattr(self, f"k_linear_{i}").bias)
            qh_lvl = q_lvl.view(q_lvl.shape[0], q_lvl.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
            kh_lvl = k_lvl.view(k_lvl.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k_lvl.shape[-2], k_lvl.shape[-1])
            weights = torch.einsum("bqnc,bnchw->bqnhw", qh_lvl * self.normalize_fact, kh_lvl)
            if mask is not None:
                weights.masked_fill_(mask[i].unsqueeze(1).unsqueeze(1), float("-inf"))
            weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
            weights = self.dropout(weights)
            out_multi_scale_maps.append(weights)

        return out_multi_scale_maps


class MultiScaleMHAttentionMapsReference(MultiScaleMHAttentionMap):
    def __init__(self, query_dim, hidden_dim, num_heads, num_levels):
        super().__init__(query_dim, hidden_dim, num_heads, num_levels)
        self.reference_point_layer = nn.Linear(query_dim + 2, hidden_dim, bias=True)

    def forward(self, q, k, mask=None):
        q = self.reference_point_layer(q)
        return super(MultiScaleMHAttentionMapsReference, self).forward(q, k, mask)


class MaskHeadSmallConvDefaultFullUpscale(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim, nheads=0, extra_feat_dim=None):
        super().__init__()
        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]

        if extra_feat_dim is None:
            extra_feat_dim = dim

        self.lay0 = torch.nn.Conv2d(dim, extra_feat_dim, 3, padding=1)
        self.gn0 = torch.nn.GroupNorm(8, extra_feat_dim)

        self.lay1 = torch.nn.Conv2d(extra_feat_dim, inter_dims[1], 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, inter_dims[1])

        self.lay2 = torch.nn.Conv2d(inter_dims[1] + nheads, inter_dims[2], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[2])

        self.lay3 = torch.nn.Conv2d(inter_dims[2] + nheads, inter_dims[3], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[3])

        self.lay4 = torch.nn.Conv2d(inter_dims[3] + nheads, inter_dims[4], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[4])

        self.lay5 = torch.nn.Conv2d(inter_dims[4], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])

        self.out_lay = torch.nn.Conv2d(inter_dims[4], 1, 3, padding=1)

        self.dim = dim

        self.adapter0 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter1 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[3], inter_dims[4], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    # x -> compressed ch backbone feats bbox_masks -> attention maps computed at each level fpn_feat -> Highest spatial res feature from the
    # backbone that in this case is not use by the transformer encoder, so we need to reduce channels
    def forward(self, compressed_backbone_feats, bbox_mask, feats, instances_per_batch, extra_feat=None):

        # Check if bbox_mask are multiscale
        multi_scale_att_maps = isinstance(bbox_mask, list)
        init_bbox_mask = bbox_mask[0] if multi_scale_att_maps else bbox_mask

        # H/64 W/64
        expanded_feats = expand_multi_length(compressed_backbone_feats, instances_per_batch)
        x = torch.cat([expanded_feats, init_bbox_mask], 1)
        x = self.lay0(x)
        x = self.gn0(x)
        x = F.relu(x)
        if extra_feat is not None:
            extra_feat = expand_multi_length(extra_feat, instances_per_batch)
            x = x + extra_feat
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)

        # H/32 W/32
        cur_fpn = self.adapter0(feats[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand_multi_length(cur_fpn, instances_per_batch)
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        if multi_scale_att_maps:
            x = torch.cat([x, bbox_mask[1]], 1)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        # H/16 W/16
        cur_fpn = self.adapter1(feats[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand_multi_length(cur_fpn, instances_per_batch)
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        if multi_scale_att_maps:
            x = torch.cat([x, bbox_mask[2]], 1)
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        # H/8 W/8
        cur_fpn = self.adapter2(feats[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand_multi_length(cur_fpn, instances_per_batch)
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        if multi_scale_att_maps:
            x = torch.cat([x, bbox_mask[3]], 1)
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        # H/4 W/4
        cur_fpn = self.adapter3(feats[3])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand_multi_length(cur_fpn, instances_per_batch)
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)

        x = self.out_lay(x)
        return x


class MaskHeadSmallConvDefaultSmallFullUpscale(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim, nheads=0, concat_on_layer=True, extra_feat_dim=None):
        super().__init__()
        inter_dims = [
            dim,
            context_dim // 2,
            context_dim // 4,
            context_dim // 8,
            context_dim // 16,
            context_dim // 64]

        layer_extra_channels = nheads if concat_on_layer else 0
        adapter_extra_channels = nheads if not concat_on_layer else 0
        if extra_feat_dim is None:
            extra_feat_dim = dim

        self.lay1 = torch.nn.Conv2d(dim, extra_feat_dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, extra_feat_dim)

        self.lay2 = torch.nn.Conv2d(extra_feat_dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])

        self.lay3 = torch.nn.Conv2d(inter_dims[1] + layer_extra_channels, inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])

        self.lay4 = torch.nn.Conv2d(inter_dims[2] + layer_extra_channels, inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])

        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])

        self.out_lay = torch.nn.Conv2d(inter_dims[4], 1, 3, padding=1)
        self.concat_on_layer = concat_on_layer
        self.extra_feat_dim = extra_feat_dim
        self.dim = dim

        self.adapter0 = torch.nn.Conv2d(fpn_dims[0] + adapter_extra_channels, extra_feat_dim, 1, )
        self.adapter1 = torch.nn.Conv2d(fpn_dims[0] + adapter_extra_channels, inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1] + adapter_extra_channels, inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    # x -> compressed ch backbone feats bbox_masks -> attention maps computed at each level fpn_feat -> Highest spatial res feature from the
    # backbone that in this case is not use by the transformer encoder, so we need to reduce channels
    def forward(self, compressed_backbone_feats, bbox_mask, fpn_feat, instances_per_batch, extra_feat=None):

        multi_scale_att_maps = isinstance(bbox_mask, list)
        init_bbox_mask = bbox_mask[0] if multi_scale_att_maps else bbox_mask

        # H/64 W/64
        expanded_feats = expand_multi_length(compressed_backbone_feats, instances_per_batch)
        x = torch.cat([expanded_feats, init_bbox_mask], 1)
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)

        # H/32 W/32
        if multi_scale_att_maps and not self.concat_on_layer:
            cur_fpn = fpn_feat[1]
            if cur_fpn.size(0) != x.size(0):
                cur_fpn = expand_multi_length(cur_fpn, instances_per_batch)
            cur_fpn = self.adapter0(torch.cat([cur_fpn, bbox_mask[1]], 1))

        else:
            cur_fpn = self.adapter0(fpn_feat[0])
            if cur_fpn.size(0) != x.size(0):
                cur_fpn = expand_multi_length(cur_fpn, instances_per_batch)

        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        # H/16 W/16
        if multi_scale_att_maps and not self.concat_on_layer:
            cur_fpn = fpn_feat[2]
            if cur_fpn.size(0) != x.size(0):
                cur_fpn = expand_multi_length(cur_fpn, instances_per_batch)
            cur_fpn = self.adapter1(torch.cat([cur_fpn, bbox_mask[2]], 1))

        else:
            cur_fpn = self.adapter1(fpn_feat[1])
            if cur_fpn.size(0) != x.size(0):
                cur_fpn = expand_multi_length(cur_fpn, instances_per_batch)

        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        if multi_scale_att_maps and self.concat_on_layer:
            x = torch.cat([x, bbox_mask[1]], 1)
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        # H/8 W/8
        if multi_scale_att_maps and not self.concat_on_layer:
            cur_fpn = fpn_feat[3]
            if cur_fpn.size(0) != x.size(0):
                cur_fpn = expand_multi_length(cur_fpn, instances_per_batch)
            cur_fpn = self.adapter2(torch.cat([cur_fpn, bbox_mask[3]], 1))

        else:
            cur_fpn = self.adapter2(fpn_feat[2])
            if cur_fpn.size(0) != x.size(0):
                cur_fpn = expand_multi_length(cur_fpn, instances_per_batch)

        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        if multi_scale_att_maps and self.concat_on_layer:
            x = torch.cat([x, bbox_mask[2]], 1)
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        # H/4 W/4
        cur_fpn = self.adapter3(fpn_feat[4])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand_multi_length(cur_fpn, instances_per_batch)
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)

        x = self.out_lay(x)
        return x


class MaskHeadDeformableConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim, nheads=0, concat_on_layer=True, extra_feat_dim=None):
        super().__init__()
        inter_dims = [
            dim,
            context_dim // 2,
            context_dim // 4,
            context_dim // 8,
            context_dim // 16,
            context_dim // 64]

        layer_extra_channels = nheads if concat_on_layer else 0
        adapter_extra_channels = nheads if not concat_on_layer else 0
        if extra_feat_dim is None:
            extra_feat_dim = dim

        self.out_lay = torch.nn.Conv2d(inter_dims[4], 1, 3, padding=1)
        self.concat_on_layer = concat_on_layer
        self.extra_feat_dim = extra_feat_dim
        self.dim = dim

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0] + adapter_extra_channels, inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1] + adapter_extra_channels, inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        self.lay1 = ModulatedDeformableConv2d(dim, extra_feat_dim)
        self.gn1 = torch.nn.GroupNorm(8, extra_feat_dim)

        self.lay2 = ModulatedDeformableConv2d(extra_feat_dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])

        self.lay3 = ModulatedDeformableConv2d(inter_dims[1] + layer_extra_channels, inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])

        self.lay4 = ModulatedDeformableConv2d(inter_dims[2] + layer_extra_channels, inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])

        self.lay5 = ModulatedDeformableConv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])

    # x -> compressed ch backbone feats bbox_masks -> attention maps computed at each level fpn_feat -> Highest spatial res feature from the
    # backbone that in this case is not use by the transformer encoder, so we need to reduce channels
    def forward(self, compressed_backbone_feats, bbox_mask, fpn_feat, instances_per_batch, extra_feat=None):

        multi_scale_att_maps = isinstance(bbox_mask, list)
        init_bbox_mask = bbox_mask[0] if multi_scale_att_maps else bbox_mask

        # H/64 W/64 or H/32 W/32
        expanded_feats = expand_multi_length(compressed_backbone_feats, instances_per_batch)
        x = torch.cat([expanded_feats, init_bbox_mask], 1)
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        if self.extra_feat_dim and extra_feat is not None:
            extra_feat = expand_multi_length(extra_feat, instances_per_batch)
            x = x + extra_feat
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        # H/16 W/16
        if multi_scale_att_maps and not self.concat_on_layer:
            cur_fpn = fpn_feat[1]
            if cur_fpn.size(0) != x.size(0):
                cur_fpn = expand_multi_length(cur_fpn, instances_per_batch)
            cur_fpn = self.adapter1(torch.cat([cur_fpn, bbox_mask[1]], 1))

        else:
            cur_fpn = self.adapter1(fpn_feat[1])
            if cur_fpn.size(0) != x.size(0):
                cur_fpn = expand_multi_length(cur_fpn, instances_per_batch)

        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        if multi_scale_att_maps and self.concat_on_layer:
            x = torch.cat([x, bbox_mask[1]], 1)
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        # H/8 W/8
        if multi_scale_att_maps and not self.concat_on_layer:
            cur_fpn = fpn_feat[2]
            if cur_fpn.size(0) != x.size(0):
                cur_fpn = expand_multi_length(cur_fpn, instances_per_batch)
            cur_fpn = self.adapter2(torch.cat([cur_fpn, bbox_mask[2]], 1))

        else:
            cur_fpn = self.adapter2(fpn_feat[2])
            if cur_fpn.size(0) != x.size(0):
                cur_fpn = expand_multi_length(cur_fpn, instances_per_batch)

        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        if multi_scale_att_maps and self.concat_on_layer:
            x = torch.cat([x, bbox_mask[2]], 1)
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        # H/4 W/4
        cur_fpn = self.adapter3(fpn_feat[3])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand_multi_length(cur_fpn, instances_per_batch)
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)

        x = self.out_lay(x)
        return x


class ModulatedDeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):
        super(ModulatedDeformableConv2d, self).__init__()

        self.padding = padding

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size * kernel_size,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size * kernel_size,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))

        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator
                                          )
        return x
