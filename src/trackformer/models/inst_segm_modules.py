import torch
import torch.nn as nn
import torch.nn.functional as F
from .detr_segmentation import expand_multi_length


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
        # level_bbox_masks = torch.cat([bbx[0] for bbx in bbox_mask], dim=1)
        x_0 = torch.cat([expand_multi_length(compressed_backbone_feats[0], instances_per_batch), bbox_mask[0]], 1)
        x_0 = self.lvl0_conv1(x_0)
        x_0 = self.lvl0_conv2(x_0)

        # H/32 W/32
        # level_bbox_masks = torch.cat([bbx[1] for bbx in bbox_mask], dim=1)
        x_1 = torch.cat([expand_multi_length(compressed_backbone_feats[1], instances_per_batch),  bbox_mask[1]], 1)
        x_1 = self.lvl1_conv(x_1)
        x_1 = x_1 + F.interpolate(x_0, size=x_1.shape[-2:], mode="nearest")
        x_1 = self.lvl1_merge(x_1)

        # H/16 W/16
        # level_bbox_masks = torch.cat([bbx[2] for bbx in bbox_mask], dim=1)
        x_2 = torch.cat([expand_multi_length(compressed_backbone_feats[2], instances_per_batch), bbox_mask[2]], 1)
        x_2 = self.lvl2_conv(x_2)
        x_2 = x_2 + F.interpolate(x_1, size=x_2.shape[-2:], mode="nearest")
        x_2 = self.lvl2_merge(x_2)

        # H/8 W/8
        # level_bbox_masks = torch.cat([bbx[3] for bbx in bbox_mask], dim=1)
        x_3 = torch.cat([expand_multi_length(compressed_backbone_feats[3], instances_per_batch), bbox_mask[3]], 1)
        x_3 = self.lvl3_conv(x_3)
        x_3 = x_3 + F.interpolate(x_2, size=x_3.shape[-2:], mode="nearest")
        x_3 = self.lvl3_merge(x_3)

        # H/4 W/4
        last_fpn = self.adapter_fpn(features[-1])
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
            # q -> object query embeddings
            # k -> encoder image feats
            q_lvl = getattr(self, f"q_linear_{i}")(q_lvl)

            # Filter shape ->  # (hidden_dim, hidden_dim, 1 , 1) corresponds to (out_channels, in_channels/groups, kH, kW) # 1-D Channels wise separated convolution
            k_lvl = F.conv2d(k_lvl, getattr(self, f"k_linear_{i}").weight.unsqueeze(-1).unsqueeze(-1), getattr(self, f"k_linear_{i}").bias)
            qh_lvl = q_lvl.view(q_lvl.shape[0], q_lvl.shape[1], self.num_heads, self.hidden_dim // self.num_heads)  # (1, 300, 8, 32)
            kh_lvl = k_lvl.view(k_lvl.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k_lvl.shape[-2],
                                k_lvl.shape[-1])  # (1, 8, 32, 19, 26)
            # Sort of batch matrix multiplication but for each the input embedings and multihead that we have: b*n*c*h*w x w*h*n*q -> b*q*n*c
            weights = torch.einsum("bqnc,bnchw->bqnhw", qh_lvl * self.normalize_fact, kh_lvl)  # (1, 300, 8, 19, 26)
            if mask is not None:
                mask_lvl = mask[i]
                weights.masked_fill_(mask_lvl.unsqueeze(1).unsqueeze(1), float("-inf"))
            weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
            weights = self.dropout(weights)  # (1, 300, 8, 19, 26)
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

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()
        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]

        self.lay0 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn0 = torch.nn.GroupNorm(8, dim)

        self.lay1 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, inter_dims[1])

        self.lay2 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[2])

        self.lay3 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[3])

        self.lay4 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
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
    def forward(self, compressed_backbone_feats, bbox_mask, feats, instances_per_batch):

        # H/64 W/64
        expanded_feats = expand_multi_length(compressed_backbone_feats, instances_per_batch)
        x = torch.cat([expanded_feats, bbox_mask], 1)
        x = self.lay0(x)
        x = self.gn0(x)
        x = F.relu(x)
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)

        # H/32 W/32
        cur_fpn = self.adapter0(feats[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand_multi_length(cur_fpn, instances_per_batch)
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)


        # H/16 W/16
        cur_fpn = self.adapter1(feats[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand_multi_length(cur_fpn, instances_per_batch)
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        # H/8 W/8
        cur_fpn = self.adapter2(feats[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand_multi_length(cur_fpn, instances_per_batch)
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
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