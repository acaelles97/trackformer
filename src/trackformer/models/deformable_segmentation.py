import random
import torch.nn as nn
import warnings
import torch
import torchvision
import torch.nn.functional as F
from ..util.misc import NestedTensor, match_name_keywords
from .matcher import HungarianMatcher
from typing import List, Dict
from torch import Tensor
ch_dict_en = {
    "/64": 256,
    "/32": 2048,
    "/16": 1024,
    "/8": 512,
    "/4": 256,
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


class DETRSegmBase(nn.Module):
    def __init__(self, only_positive_matches: bool, freeze_detr: bool, trainable_params: List, top_k_predictions: int, matcher: HungarianMatcher,
                 mask_head_used_features: List[List[str]], att_maps_used_res: List[str], use_deformable_conv: bool):

        if freeze_detr:
            for n, p in self.named_parameters():
                if not match_name_keywords(n, trainable_params):
                    p.requires_grad_(False)

        self.only_positive = only_positive_matches
        self.top_k_predictions = top_k_predictions
        self.matcher = matcher
        self.mask_head_used_features = mask_head_used_features
        self.att_maps_used_res = att_maps_used_res
        self._sanity_check()

        feats_dims = self._get_mask_head_dims()
        hidden_dim, nheads = self.transformer.d_model, self.transformer.nhead
        self.bbox_attention = MultiScaleMHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0, num_levels=len(self.att_maps_used_res))
        self.mask_head = MaskHeadConv(hidden_dim, feats_dims, nheads, use_deformable_conv, self.att_maps_used_res)

    def _sanity_check(self):
        init_mask_head_res, init_att_map_res = self.mask_head_used_features[0][0], self.att_maps_used_res[0]
        assert init_mask_head_res == init_att_map_res, f"Starting resolution for the mask_head_used features and att_maps_used_res has to be " \
                                                       f"the same. Got {init_mask_head_res} and {init_att_map_res} respectively"
        parent_class = [base.__name__ for base in self.__class__.__bases__]
        for cls in parent_class:
            if cls == "DETR":
                assert self.mask_head_used_features == [['/32', 'compressed_backbone'], ['/16', 'backbone'], ['/8', 'backbone'], ['/4', 'backbone']], \
                    "Only the following mask_head_used_features are available for DeTR: " \
                    "[['/32','compressed_backbone'], ['/16','backbone'], ['/8','backbone'], ['/4','backbone']]"
                assert self.att_maps_used_res == ['/32'], "Only the following mask head features are available for DeTR"

    def _get_mask_head_dims(self):
        feats_dims = []
        for res, name in self.mask_head_used_features[1:]:
            if name == "backbone":
                feats_dims.append(ch_dict_en[res])
            else:
                feats_dims.append(256)
        return feats_dims

    @staticmethod
    def get_src_permutation_idx(indices: List[Tensor]):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, src in enumerate(indices)])
        src_idx = torch.cat([src for src in indices])
        return batch_idx, src_idx

    @staticmethod
    def tmp_batch_fill(num_embd: int, matched_indices: List[Tensor]):
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

    def _get_matched_with_filled_embeddings(self, indices: List[Tensor], hs: Tensor):
        instances_per_batch = [idx[0].shape[0] for idx in indices]
        filled_indices = self.tmp_batch_fill(hs.shape[2], indices)
        num_filled_instances = len(filled_indices[0])
        matched_indices = self.get_src_permutation_idx(filled_indices)
        matched_embeddings = hs[-1][matched_indices].view(hs.shape[1], num_filled_instances, hs.shape[-1])
        return matched_embeddings, instances_per_batch

    def _get_features_for_mask_head(self, backbone_feats: List[Tensor], srcs: List[Tensor], memories: List[Tensor]):
        features_used = []
        for res, feature_type in self.mask_head_used_features:
            if feature_type == "backbone":
                if res == "/64":
                    warnings.warn("/64 feature map is only generated for encoded and compressed backbone feats. Using the compressed one")
                    features_used.append(srcs[res_to_idx[res]])
                else:
                    features_used.append(backbone_feats[backbone_res_to_idx[res]].tensors)
            elif feature_type == "compressed_backbone":
                if res == "/4":
                    warnings.warn("/4 feature map is only generated for backbone. Using backbone")
                    features_used.append(backbone_feats[backbone_res_to_idx[res]].tensors)
                else:
                    features_used.append(srcs[res_to_idx[res]])

            elif feature_type == "encoded":
                if res == "/4":
                    warnings.warn("/4 feature map is only generated for backbone. Using backbone")
                    features_used.append(backbone_feats[backbone_res_to_idx[res]].tensors)
                else:
                    features_used.append(memories[res_to_idx[res]])
            else:
                raise ValueError(
                    f"Selected feature type {feature_type} is not available. Available ones: [backbone, compressed_backbone, encoded]")

        return features_used

    # Implements mask computation from forward output for inference taking only into account top k predictions
    def _predict_masks(self, out: Dict[str, Tensor], hs: Tensor, memories_att_map: List[Tensor], masks_att_map: List[Tensor], mask_head_feats: List[Tensor]):
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
        out, targets, backbone_feats, memories, hs, srcs, masks = super().forward(samples, targets)
        generate_predictions = targets[0]["generate_predictions"]

        if not isinstance(memories, list):
            memories_att_map, masks_att_map = [memories], [masks]
        else:
            memories_att_map = [memories[res_to_idx[res]] for res in self.att_maps_used_res]
            masks_att_map = [masks[res_to_idx[res]] for res in self.att_maps_used_res]

        if self.only_positive:
            outputs_without_aux = {k: v for k, v in out.items() if k != 'aux_outputs' and k != 'enc_outputs' and k != 'hs_embed'}
            indices = self.matcher(outputs_without_aux, targets)
            out["indices"] = indices
            matched_embeddings, instances_per_batch = self._get_matched_with_filled_embeddings(indices, hs)
            indices_to_pick = [torch.arange(0, num_instances) for num_instances in instances_per_batch]
            indices_to_pick = self.get_src_permutation_idx(indices_to_pick)
        else:
            matched_embeddings = hs[-1]
            instances_per_batch = matched_embeddings.shape[1]
            indices_to_pick = [torch.arange(0, matched_embeddings.shape[1]) for _ in range(matched_embeddings.shape[0])]
            indices_to_pick = self.get_src_permutation_idx(indices_to_pick)

        bbox_masks = self.bbox_attention(matched_embeddings, memories_att_map, mask=masks_att_map)
        bbox_masks = [bbox_mask[indices_to_pick] for bbox_mask in bbox_masks]

        # Mask used for loss computation
        if not isinstance(srcs, list):
            mask_head_feats = [srcs, backbone_feats[2].tensors, backbone_feats[1].tensors, backbone_feats[0].tensors]
        else:
            mask_head_feats = self._get_features_for_mask_head(backbone_feats, srcs, memories)

        seg_masks = self.mask_head(mask_head_feats, bbox_masks, instances_per_batch=instances_per_batch)

        if self.only_positive:
            out["pred_masks"] = seg_masks
        else:
            seg_masks = seg_masks.view(hs.shape[1], instances_per_batch, seg_masks.shape[-2], seg_masks.shape[-1])
            out["pred_masks"] = seg_masks

        # Masks used for inference/validation
        if not self.training or generate_predictions:
            if self.only_positive:
                out["inference_masks"] = self._predict_masks(out, hs, memories_att_map, masks_att_map, mask_head_feats)
            else:
                out["inference_masks"] = seg_masks

        return out, targets, backbone_feats, memories, hs


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
        offset = self.offset_conv(x)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        x = torchvision.ops.deform_conv2d(input=x, offset=offset, weight=self.regular_conv.weight, bias=self.regular_conv.bias,
                                          padding=self.padding, mask=modulator)
        return x


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        nn.init.kaiming_uniform_(self.weight, a=1)
        nn.init.constant_(self.bias, 0)


def expand_multi_length(tensor, lengths):
    if isinstance(lengths, list):
        tensors = []
        for idx, length_to_repeat in enumerate(lengths):
            tensors.append(tensor[idx].unsqueeze(0).repeat(1, int(length_to_repeat), 1, 1, 1).flatten(0, 1))
        return torch.cat(tensors, dim=0)
    else:
        return tensor.unsqueeze(1).repeat(1, int(lengths), 1, 1, 1).flatten(0, 1)


class MultiScaleMHAttentionMap(nn.Module):

    def __init__(self, query_dim, hidden_dim, num_heads, num_levels, dropout=0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        for i in range(num_levels):
            layer_name = "" if i == 0 else f"_{i}"
            setattr(self, f"q_linear{layer_name}", nn.Linear(query_dim, hidden_dim, bias=bias))
            setattr(self, f"k_linear{layer_name}", nn.Linear(query_dim, hidden_dim, bias=bias))
            nn.init.zeros_(getattr(self, f"k_linear{layer_name}").bias)
            nn.init.zeros_(getattr(self, f"q_linear{layer_name}").bias)
            nn.init.xavier_uniform_(getattr(self, f"k_linear{layer_name}").weight)
            nn.init.xavier_uniform_(getattr(self, f"q_linear{layer_name}").weight)

        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def _check_input(self, k, mask):
        assert len(k) == self.num_levels
        if mask is not None:
            assert len(mask) == self.num_levels

    def forward(self, q, k, mask=None):
        self._check_input(k, mask)

        out_multi_scale_maps = []

        for i, k_lvl in enumerate(k):
            layer_name = "" if i == 0 else f"_{i}"
            q_lvl = q
            q_lvl = getattr(self, f"q_linear{layer_name}")(q_lvl)
            k_lvl = F.conv2d(k_lvl, getattr(self, f"k_linear{layer_name}").weight.unsqueeze(-1).unsqueeze(-1),
                             getattr(self, f"k_linear{layer_name}").bias)
            qh_lvl = q_lvl.view(q_lvl.shape[0], q_lvl.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
            kh_lvl = k_lvl.view(k_lvl.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k_lvl.shape[-2], k_lvl.shape[-1])
            weights = torch.einsum("bqnc,bnchw->bqnhw", qh_lvl * self.normalize_fact, kh_lvl)
            if mask is not None:
                weights.masked_fill_(mask[i].unsqueeze(1).unsqueeze(1), float("-inf"))
            weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
            weights = self.dropout(weights)
            out_multi_scale_maps.append(weights)

        return out_multi_scale_maps


class MaskHeadConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, nheads, use_deformable_conv, multi_scale_att_maps):
        super().__init__()

        num_levels = len(fpn_dims) + 1
        out_dims = [dim // (2 ** exp) for exp in range(num_levels + 1)]
        in_dims = [dim // (2 ** exp) for exp in range(num_levels + 1)]
        for i in range(len(multi_scale_att_maps)):
            in_dims[i] += nheads

        self.multi_scale_att_maps = len(multi_scale_att_maps) > 1
        conv_layer = ModulatedDeformableConv2d if use_deformable_conv else Conv2d

        self.lay1 = conv_layer(in_dims[0], in_dims[0], 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, in_dims[0])

        self.lay2 = conv_layer(in_dims[0], out_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, out_dims[1])

        for i in range(1, num_levels):
            setattr(self, f"lay{i + 2}", conv_layer(in_dims[i], out_dims[i + 1], 3, padding=1))
            setattr(self, f"gn{i + 2}", torch.nn.GroupNorm(8, out_dims[i + 1]))
            setattr(self, f"adapter{i}", Conv2d(fpn_dims[i - 1], out_dims[i], 1, padding=0))

        self.out_lay = Conv2d(out_dims[i + 1], 1, 3, padding=1)

    def forward(self, features, bbox_mask, instances_per_batch):

        expanded_feats = expand_multi_length(features[0], instances_per_batch)
        x = torch.cat([expanded_feats, bbox_mask[0]], 1)
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        for lvl, feature in enumerate(features[1:]):
            cur_fpn = getattr(self, f"adapter{lvl + 1}")(feature)
            cur_fpn = expand_multi_length(cur_fpn, instances_per_batch)
            x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
            if self.multi_scale_att_maps and lvl + 1 < len(bbox_mask):
                x = torch.cat([x, bbox_mask[lvl + 1]], 1)
            x = getattr(self, f"lay{lvl + 3}")(x)
            x = getattr(self, f"gn{lvl + 3}")(x)
            x = F.relu(x)

        x = self.out_lay(x)
        return x
