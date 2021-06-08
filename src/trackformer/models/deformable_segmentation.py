import copy
import random
from ..util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                         get_world_size, interpolate, match_name_keywords,
                         is_dist_avail_and_initialized, dice_loss, sigmoid_focal_loss)

from .deformable_detr import DeformableDETR
from .inst_segm_modules import *
from .detr_segmentation import MaskHeadSmallConvDefault, MHAttentionMapDefault
from .ops.modules import MSDeformAttn, MSDeformAttnPytorch
import warnings

ch_dict_en = {
    "64": 256,
    "32": 2048,
    "16": 1024,
    "8": 512,
    "4": 256,
}

res_to_idx = {
    "/64": 3,
    "/32": 2,
    "/16": 1,
    "/8": 0,
}

backbone_res_to_idx = {
    "/32": 3,
    "/16": 2,
    "/8": 1,
    "/4": 0,
}

class BaseSegm(nn.Module):
    def __init__(self, freeze_detr, trainable_params, used_feats, top_k_predictions, matcher, only_positive):

        super().__init__()

        if freeze_detr:
            for n, p in self.named_parameters():
                if not match_name_keywords(n, trainable_params):
                    p.requires_grad_(False)

        self.only_positive = only_positive
        self.top_k_predictions = top_k_predictions
        self.matcher = matcher
        self.used_feats = [("/64", "src"), ("/32", "encoded"), ("/16", "encoded"), ("/8", "encoded")]

        feats_dims = [256, 256, 256] if self.use_encoded_feats else [1024, 512, 256]
        self.bbox_attention = MultiScaleMHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0, num_levels=self.num_levels)

    def get_top_k_indices(self, outputs):
        out_logits = outputs['pred_logits']
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.top_k_predictions, dim=1)
        outputs["top_k_values"], outputs["top_k_indexes"] = topk_values, topk_indexes
        return topk_indexes // out_logits.shape[2]

    @staticmethod
    def get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, src in enumerate(indices)])
        src_idx = torch.cat([src for src in indices])
        return batch_idx, src_idx

    @staticmethod
    def tmp_batch_fill(num_embd, matched_indices):
        new_indices = []
        max_num = max([idx[0].shape[0] for idx in matched_indices])
        all_pos = set(range(0, num_embd))
        for idx, (embd_idxs, _) in enumerate(matched_indices):
            num_to_fill = max_num - len(embd_idxs)
            if num_to_fill > 0:
                batch_ids = set(embd_idxs.tolist())
                unmatched_embds = random.choices(list(all_pos.difference(batch_ids)), k=num_to_fill)
                new_embd_idxs = torch.cat([embd_idxs, torch.tensor(unmatched_embds, dtype=torch.int64)])
                new_indices.append(new_embd_idxs)
            else:
                new_indices.append(embd_idxs)

        return new_indices

    def get_matched_with_filled_embeddings(self, indices, hs):
        instances_per_batch = [idx[0].shape[0] for idx in indices]
        filled_indices = self.tmp_batch_fill(hs.shape[2], indices)
        num_filled_instances = len(filled_indices[0])
        matched_indices = self.get_src_permutation_idx(filled_indices)
        matched_embeddings = hs[-1][matched_indices].view(hs.shape[1], num_filled_instances, hs.shape[-1])
        return matched_embeddings, instances_per_batch


    def _get_features_for_mask_head(self, backbone_feats, srcs, memories):
        features_used = []
        for res, feature_type in self.used_feats:
            if feature_type == "backbone":
                if res == "/64":
                    warnings.warn("/64 feature map is only generated for encoded and compressed backbone feats. Using the compressed one")
                    features_used.append(srcs[res_to_idx[res]])
                else:
                    features_used.append(backbone_feats[backbone_res_to_idx[res]].tensors)
            elif feature_type == "compressed_backbone":
                features_used.append(srcs[res_to_idx[res]])

            elif feature_type == "encoded":
                features_used.append(memories[res_to_idx[res]])
            else:
                raise ValueError(f"Selected feature type {feature_type} is not available. Available ones: [backbone, compressed_backbone, encoded]")

        return features_used

    # Implements mask computation from forward output for inference taking only into account top k predictions
    def _predict_masks(self, out, hs, memories_att_map, masks_att_map, mask_head_feats):
        out_logits = out['pred_logits']
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.top_k_predictions, dim=1)
        out["top_k_values"], out["top_k_indexes"] = topk_values, topk_indexes
        indices = topk_indexes // out_logits.shape[2]
        objs_embeddings = torch.gather(hs[-1], 1, indices.unsqueeze(-1).repeat(1, 1, hs[-1].shape[-1]))

        bbox_masks = self.bbox_attention(objs_embeddings, memories_att_map, mask=masks_att_map)
        bbox_masks = [bbox_mask.flatten(0, 1) for bbox_mask in bbox_masks]

        out_masks = self.mask_head(mask_head_feats, bbox_masks, instances_per_batch=self.top_k_predictions)
        outputs_seg_masks = out_masks.view(hs.shape[1], self.top_k_predictions, out_masks.shape[-2], out_masks.shape[-1])

        return outputs_seg_masks

    def forward(self, samples: NestedTensor, targets: list = None):
        out, targets, backbone_feats, memories, hs, srcs, pos_embd, masks, inter_references, query_embed = super().forward(samples, targets)

        if self.only_positive:
            outputs_without_aux = {k: v for k, v in out.items() if k != 'aux_outputs' and k != 'enc_outputs' and k != 'hs_embed'}
            indices = self.matcher(outputs_without_aux, targets)
            out["indices"] = indices

            memories_att_map = [memories[res_to_idx[res]] for res, _ in self.used_feats]
            masks_att_map = [masks[res_to_idx[res]] for res, _ in self.used_res]

            matched_embeddings, instances_per_batch = self.get_matched_with_filled_embeddings(indices, hs)
            bbox_masks = self.bbox_attention(matched_embeddings, memories_att_map, mask=masks_att_map)
            indices_to_pick = [torch.arange(0, num_instances) for num_instances in instances_per_batch]
            indices_to_pick = self.get_src_permutation_idx(indices_to_pick)
            bbox_masks = [bbox_mask[indices_to_pick] for bbox_mask in bbox_masks]

            mask_head_feats = self.get_features_for_mask_head(backbone_feats, srcs, memories)
            # Mask used for loss computation
            out["pred_masks"] = self.mask_head(mask_head_feats, bbox_masks, instances_per_batch=instances_per_batch)

            generate_predictions = targets[0]["generate_predictions"]
            # Masks used for inference/validation
            if not self.training or generate_predictions:
                out["inference_masks"] = self.predict_masks(out, hs, memories_att_map, masks_att_map, mask_head_feats)

        else:
            instances_per_batch = hs.shape[2]
            # Check if it comes from DeTr or Def DeTr
            if isinstance(memories, list):
                batch_size = memories[0].shape[0]

                src, mask, memory = srcs[-3], masks[-3], memory[-3]

                # fpns = [memory[2], memory[1], memory[0]]
                fpns = [features[-1].tensors, features[-2].tensors, features[-3].tensors, features[-4].tensors]

            else:
                src, mask = features[-1].decompose()
                batch_size = src.shape[0]
                src = self.input_proj(src)
                fpns = [None, features[2].tensors, features[1].tensors, features[0].tensors]

            # FIXME h_boxes takes the last one computed, keep this in mind
            bbox_mask = self.bbox_attention(hs[-1], memory, mask=mask)
            bbox_mask = bbox_mask.flatten(0, 1)

            seg_masks = self.mask_head(src, bbox_mask, fpns, instances_per_batch=instances_per_batch)
            outputs_seg_masks = seg_masks.view(
                batch_size, hs.shape[2], seg_masks.shape[-2], seg_masks.shape[-1])

            out["pred_masks"] = outputs_seg_masks

            return out, targets, features, memory, hs, srcs, pos, masks, inter_references, query_embed


        return out, targets




class ModulatedDeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(ModulatedDeformableConv2d, self).__init__()

        self.padding = padding

        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride,
                                     padding=self.padding, bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels, 1 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride,
                                        padding=self.padding, bias=True)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=self.padding, bias=bias)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))

        x = torchvision.ops.deform_conv2d(input=x, offset=offset, weight=self.regular_conv.weight, bias=self.regular_conv.bias,
                                          padding=self.padding, mask=modulator)
        return x


class Conv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        nn.init.kaiming_uniform_(self.weight, a=1)
        nn.init.constant_(self.bias, 0)


class MaskHeadConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim, nheads, use_deformable_conv, concat_on_layer):
        super().__init__()
        inter_dims = [
            dim,
            context_dim // 2,
            context_dim // 4,
            context_dim // 8,
            context_dim // 16,
            context_dim // 64]

        self.concat_on_layer = concat_on_layer
        layer_extra_channels = nheads if self.concat_on_layer else 0
        adapter_extra_channels = nheads if not self.concat_on_layer else 0
        conv_layer = ModulatedDeformableConv2d if use_deformable_conv else Conv2d

        self.lay1 = conv_layer(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, dim)

        self.lay2 = conv_layer(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])

        self.lay3 = conv_layer(inter_dims[1] + layer_extra_channels, inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])

        self.lay4 = conv_layer(inter_dims[2] + layer_extra_channels, inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])

        self.lay5 = conv_layer(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])

        self.out_lay = Conv2d(inter_dims[4], 1, 3, padding=1)

        self.adapter1 = Conv2d(fpn_dims[0] + adapter_extra_channels, inter_dims[1], 1)
        self.adapter2 = Conv2d(fpn_dims[1] + adapter_extra_channels, inter_dims[2], 1)
        self.adapter3 = Conv2d(fpn_dims[2], inter_dims[3], 1)

    def forward(self, features, bbox_mask, instances_per_batch):

        multi_scale_att_maps = isinstance(bbox_mask, list)
        init_bbox_mask = bbox_mask[0] if multi_scale_att_maps else bbox_mask

        # H/32 W/32
        expanded_feats = expand_multi_length(features[0], instances_per_batch)
        x = torch.cat([expanded_feats, init_bbox_mask], 1)
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        # H/16 W/16
        if multi_scale_att_maps and not self.concat_on_layer:
            cur_fpn = expand_multi_length(features[1], instances_per_batch)
            cur_fpn = self.adapter1(torch.cat([cur_fpn, bbox_mask[1]], 1))
        else:
            cur_fpn = self.adapter1(features[1])
            cur_fpn = expand_multi_length(cur_fpn, instances_per_batch)

        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        if multi_scale_att_maps and self.concat_on_layer:
            x = torch.cat([x, bbox_mask[1]], 1)
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        # H/8 W/8
        if multi_scale_att_maps and not self.concat_on_layer:
            cur_fpn = expand_multi_length(features[2], instances_per_batch)
            cur_fpn = self.adapter2(torch.cat([cur_fpn, bbox_mask[2]], 1))
        else:
            cur_fpn = self.adapter2(features[2])
            cur_fpn = expand_multi_length(cur_fpn, instances_per_batch)

        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        if multi_scale_att_maps and self.concat_on_layer:
            x = torch.cat([x, bbox_mask[2]], 1)
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        # H/4 W/4
        cur_fpn = self.adapter3(features[3])
        cur_fpn = expand_multi_length(cur_fpn, instances_per_batch)
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)

        x = self.out_lay(x)
        return x


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
