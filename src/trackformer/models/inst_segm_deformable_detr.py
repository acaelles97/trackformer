import copy
import random
from ..util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                         get_world_size, interpolate, match_name_keywords,
                         is_dist_avail_and_initialized, dice_loss, sigmoid_focal_loss)

from .deformable_detr import DeformableDETR
from .inst_segm_modules import *
from .detr_segmentation import MaskHeadSmallConvDefault, MHAttentionMapDefault
from .ops.modules import MSDeformAttn, MSDeformAttnPytorch


class DefDETRInstanceSeg(DeformableDETR):
    def __init__(self, mask_kwargs, detr_kwargs):
        super().__init__(**detr_kwargs)
        trainable_params = mask_kwargs['trainable_params']

        if mask_kwargs["freeze_detr"]:
            for n, p in self.named_parameters():
                if not match_name_keywords(n, trainable_params):
                    p.requires_grad_(False)

        # if mask_kwargs["freeze_detr"]:
        #     for n, p in self.named_parameters():
        #         p.requires_grad_(False)

        self.top_k_predictions = mask_kwargs["top_k_predictions"]
        self.matcher = mask_kwargs["matcher"]
        self.use_encoded_feats = mask_kwargs["use_encoded_feats"]
        self.fill_batch = mask_kwargs["fill_batch"]
        self.batch_mode = mask_kwargs["batch_mode"]
        assert not (self.fill_batch and self.batch_mode), "Fill batch and batch mode can not be set True at the same time!"

    def get_top_k_indices(self, outputs):
        out_logits = outputs['pred_logits']
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.top_k_predictions, dim=1)
        outputs["top_k_values"] = topk_values
        outputs["top_k_indexes"] = topk_indexes
        return topk_indexes // out_logits.shape[2]

    def get_src_permutation_idx(self, indices):
        # permute predictions following indices
        if isinstance(indices[0], tuple):
            batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
            src_idx = torch.cat([src for (src, _) in indices])
        else:
            batch_idx = torch.cat([torch.full_like(src, i) for i, src in enumerate(indices)])
            src_idx = torch.cat([src for src in indices])
        return batch_idx, src_idx

    @staticmethod
    def fill_batch_with_random_samples(num_embd, matched_indices, targets):
        new_indices = []
        max_num = max([idx[0].shape[0] for idx in matched_indices])
        all_pos = set(range(0, num_embd))
        for idx, (embd_idxs, tgt_idxs) in enumerate(matched_indices):
            num_to_fill = max_num - len(embd_idxs)
            if num_to_fill > 0:
                batch_ids = set(embd_idxs.tolist())
                unmatched_embds = random.choices(list(all_pos.difference(batch_ids)), k=num_to_fill)
                new_embd_idxs = torch.cat([embd_idxs, torch.tensor(unmatched_embds, dtype=torch.int64)])
                new_tgt_idxs = torch.cat([tgt_idxs, torch.arange(tgt_idxs.shape[0], max_num, dtype=torch.int64)])
                tgt_size = targets[idx]["size"]
                empty_masks = torch.zeros((num_to_fill, tgt_size[0], tgt_size[1]), dtype=torch.bool, device=targets[idx]["masks"].device)
                targets[idx]["masks"] = torch.cat([targets[idx]["masks"], empty_masks])
                new_indices.append((new_embd_idxs, new_tgt_idxs))

            else:
                new_indices.append((embd_idxs, tgt_idxs))

        return new_indices

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

    def prepare_batch_fill_random_embeddings(self, out, hs, indices, targets):
        filled_indices = self.fill_batch_with_random_samples(hs.shape[2], indices, targets)
        out["filled_indices"] = filled_indices
        instances_per_batch = len(filled_indices[0][0])
        matched_indices = self.get_src_permutation_idx(filled_indices)
        matched_embeddings = hs[-1][matched_indices].view(hs.shape[1], instances_per_batch, hs.shape[-1])
        return matched_embeddings, instances_per_batch

    def prepare_tmp_batch_fill(self, indices, hs):
        instances_per_batch = [idx[0].shape[0] for idx in indices]
        filled_indices = self.tmp_batch_fill(hs.shape[2], indices)
        num_filled_instances = len(filled_indices[0])
        matched_indices = self.get_src_permutation_idx(filled_indices)
        matched_embeddings = hs[-1][matched_indices].view(hs.shape[1], num_filled_instances, hs.shape[-1])
        return matched_embeddings, instances_per_batch

    def forward(self, samples: NestedTensor, targets: list = None):
        out, targets, features, memories, hs, srcs, pos_embd, masks, inter_references, query_embed = super().forward(samples, targets)

        outputs_without_aux = {k: v for k, v in out.items() if k != 'aux_outputs' and k != 'enc_outputs' and k != 'hs_embed'}
        indices = self.matcher(outputs_without_aux, targets)
        out["indices"] = indices
        return out, targets, features, memories, srcs, masks, hs, pos_embd, inter_references, query_embed

    def get_top_k_embeddings(self, out, hs):
        indices = self.get_top_k_indices(out)
        objs_embeddings = torch.gather(hs[-1], 1, indices.unsqueeze(-1).repeat(1, 1, hs[-1].shape[-1]))
        return objs_embeddings


class DefDETRInstanceSegTopK(DefDETRInstanceSeg):
    def __init__(self, mask_kwargs, detr_kwargs):
        super().__init__(mask_kwargs, detr_kwargs)

        self.res_lvl = mask_kwargs["attention_map_lvl"]
        hidden_dim, nheads = self.transformer.d_model, self.transformer.nhead

        self.bbox_attention = MHAttentionMapDefault(hidden_dim, hidden_dim, nheads, dropout=0)

        feats_dims = [256, 256, 256] if self.use_encoded_feats else [1024, 512, 256]
        self.mask_head = MaskHeadSmallConvDefault(hidden_dim + nheads, feats_dims, hidden_dim)

    def forward(self, samples: NestedTensor, targets: list = None):

        out, targets, features, memories, srcs, masks, hs, pos_embd, inter_references, query_embed = super().forward(samples, targets)

        indices = out["indices"]
        memories, srcs, masks = list(reversed(memories)), list(reversed(srcs)), list(reversed(masks))
        backbone_feats = list(reversed([feat.tensors for feat in features]))

        if self.fill_batch:
            matched_embeddings, instances_per_batch = self.prepare_batch_fill_random_embeddings(out, hs, indices, targets)
            bbox_masks = self.bbox_attention(matched_embeddings, memories[self.res_lvl], mask=masks[self.res_lvl])
            bbox_masks = bbox_masks.flatten(0, 1)

        elif self.batch_mode:
            matched_embeddings, instances_per_batch = self.prepare_tmp_batch_fill(indices, hs)
            bbox_masks = self.bbox_attention(matched_embeddings, memories[self.res_lvl], mask=masks[self.res_lvl])
            indices_to_pick = [torch.arange(0, num_instances) for num_instances in instances_per_batch]
            indices_to_pick = self.get_src_permutation_idx(indices_to_pick)
            bbox_masks = bbox_masks[indices_to_pick]

        else:
            instances_per_batch = [idx[0].shape[0] for idx in indices]
            bbox_masks = []
            # We have different num of masks to compute for each batch so we need to split computation
            for idx, (embd_idxs, _) in enumerate(indices):
                matched_embeddings = hs[-1, idx, embd_idxs].unsqueeze(0)
                batch_memories = [mem[idx].unsqueeze(0) for mem in memories]
                batch_masks = [mask[idx].unsqueeze(0) for mask in masks]
                bbox_mask = self.bbox_attention(matched_embeddings, batch_memories[self.res_lvl], mask=batch_masks[self.res_lvl])
                bbox_masks.append(bbox_mask)

            bbox_masks = torch.cat(bbox_masks, dim=1).squeeze(0)

        features = [memories[1], memories[2], memories[3], backbone_feats[-1]] if self.use_encoded_feats else backbone_feats
        seg_masks = self.mask_head(srcs[self.res_lvl], bbox_masks, features, instances_per_batch=instances_per_batch)

        # Compute ouput mask for loss prediction
        out["pred_masks"] = seg_masks

        generate_predictions = targets[0]["generate_predictions"]
        # Compute Inference masks for later use when doing validation
        if not self.training or generate_predictions:
            out["inference_masks"] = self.predict_masks(out, features, srcs, memories, masks, hs)

        return out, targets, None

    # Implements mask computation from forward output for inference taking only into account top k predictions
    def predict_masks(self, out, features, srcs, memories, masks, hs):
        objs_embeddings = self.get_top_k_embeddings(out, hs)

        bbox_masks = self.bbox_attention(objs_embeddings, memories[self.res_lvl], mask=masks[self.res_lvl])
        bbox_masks = bbox_masks.flatten(0, 1)

        out_masks = self.mask_head(srcs[self.res_lvl], bbox_masks, features, instances_per_batch=self.top_k_predictions)
        outputs_seg_masks = out_masks.view(hs.shape[1], self.top_k_predictions, out_masks.shape[-2], out_masks.shape[-1])

        return outputs_seg_masks


class DefDETRInstanceSegTopKMutliScaleAttMaps(DefDETRInstanceSeg):
    def __init__(self, mask_kwargs, detr_kwargs):
        super().__init__(mask_kwargs, detr_kwargs)

        self.res_lvl = mask_kwargs["attention_map_lvl"]
        hidden_dim, nheads = self.transformer.d_model, self.transformer.nhead

        self.num_levels = 3
        self.bbox_attention = MultiScaleMHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0, num_levels=self.num_levels)
        feats_dims = [256, 256, 256] if self.use_encoded_feats else [1024, 512, 256]
        self.mask_head = MaskHeadSmallConvDefault(hidden_dim + nheads, feats_dims, hidden_dim, nheads=nheads)

    def compute_bbox_att_map(self, memories, masks, indices, hs):
        memories_att_map = [memories[0], memories[2], memories[3]] if self.res_lvl == 0 else [memories[1], memories[2], memories[3]]
        masks_att_map = [masks[0], masks[2], masks[3]] if self.res_lvl == 0 else [masks[1], masks[2], masks[3]]
        matched_embeddings, instances_per_batch = self.prepare_tmp_batch_fill(indices, hs)
        bbox_masks = self.bbox_attention(matched_embeddings, memories_att_map, mask=masks_att_map)
        indices_to_pick = [torch.arange(0, num_instances) for num_instances in instances_per_batch]
        indices_to_pick = self.get_src_permutation_idx(indices_to_pick)
        bbox_masks = [bbox_mask[indices_to_pick] for bbox_mask in bbox_masks]
        return bbox_masks, instances_per_batch

    def forward(self, samples: NestedTensor, targets: list = None):
        out, targets, features, memories, srcs, masks, hs, pos_embd, inter_references, query_embed = super().forward(samples, targets)

        indices = out["indices"]
        memories, srcs, masks = list(reversed(memories)), list(reversed(srcs)), list(reversed(masks))
        backbone_feats = list(reversed([feat.tensors for feat in features]))

        bbox_masks, instances_per_batch = self.compute_bbox_att_map(memories, masks, indices, hs)
        features = [memories[1], memories[2], memories[3], backbone_feats[-1]] if self.use_encoded_feats else backbone_feats
        seg_masks = self.mask_head(srcs[self.res_lvl], bbox_masks, features, instances_per_batch=instances_per_batch)

        # Compute ouput mask for loss prediction
        out["pred_masks"] = seg_masks

        generate_predictions = targets[0]["generate_predictions"]
        # Compute Inference masks for later use when doing validation
        if not self.training or generate_predictions:
            out["inference_masks"] = self.predict_masks(out, features, srcs, memories, masks, hs)

        return out, targets, None

    # Implements mask computation from forward output for inference taking only into account top k predictions
    def predict_masks(self, out, features, srcs, memories, masks, hs):
        objs_embeddings = self.get_top_k_embeddings(out, hs)

        memories_att_map = [memories[0], memories[2], memories[3]] if self.res_lvl == 0 else [memories[1], memories[2], memories[3]]
        masks_att_map = [masks[0], masks[2], masks[3]] if self.res_lvl == 0 else [masks[1], masks[2], masks[3]]
        bbox_masks = self.bbox_attention(objs_embeddings, memories_att_map, mask=masks_att_map)
        bbox_masks = [bbox_mask.flatten(0, 1) for bbox_mask in bbox_masks]

        out_masks = self.mask_head(srcs[0], bbox_masks, features, instances_per_batch=self.top_k_predictions)
        outputs_seg_masks = out_masks.view(hs.shape[1], self.top_k_predictions, out_masks.shape[-2], out_masks.shape[-1])

        return outputs_seg_masks


class DefDETRInstanceSegTopKMutliScaleAttMapAdapterConcat(DefDETRInstanceSegTopKMutliScaleAttMaps):
    def __init__(self, mask_kwargs, detr_kwargs):
        super().__init__(mask_kwargs, detr_kwargs)

        hidden_dim, nheads = self.transformer.d_model, self.transformer.nhead
        feats_dims = [256, 256, 256] if self.use_encoded_feats else [1024, 512, 256]
        self.mask_head = MaskHeadSmallConvDefault(hidden_dim + nheads, feats_dims, hidden_dim, nheads=nheads, concat_on_layer=False)


class DefDETRInstanceSegTopKMutliScaleAttMapsOnlyEncodedFeats(DefDETRInstanceSeg):
    def __init__(self, mask_kwargs, detr_kwargs):
        super().__init__(mask_kwargs, detr_kwargs)

        self.res_lvl = mask_kwargs["attention_map_lvl"]
        hidden_dim, nheads = self.transformer.d_model, self.transformer.nhead

        self.num_levels = 3
        self.bbox_attention = MultiScaleMHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0, num_levels=self.num_levels)
        feats_dims = [256, 256, 256] if self.use_encoded_feats else [1024, 512, 256]
        self.mask_head = MaskHeadSmallConvDefault(hidden_dim + nheads, feats_dims, hidden_dim, nheads=nheads, concat_on_layer=True)

    def compute_bbox_att_map(self, memories, masks, indices, hs):
        memories_att_map = [memories[0], memories[2], memories[3]] if self.res_lvl == 0 else [memories[1], memories[2], memories[3]]
        masks_att_map = [masks[0], masks[2], masks[3]] if self.res_lvl == 0 else [masks[1], masks[2], masks[3]]
        matched_embeddings, instances_per_batch = self.prepare_tmp_batch_fill(indices, hs)
        bbox_masks = self.bbox_attention(matched_embeddings, memories_att_map, mask=masks_att_map)
        indices_to_pick = [torch.arange(0, num_instances) for num_instances in instances_per_batch]
        indices_to_pick = self.get_src_permutation_idx(indices_to_pick)
        bbox_masks = [bbox_mask[indices_to_pick] for bbox_mask in bbox_masks]
        return bbox_masks, instances_per_batch

    def forward(self, samples: NestedTensor, targets: list = None):
        out, targets, features, memories, srcs, masks, hs, pos_embd, inter_references, query_embed = super().forward(samples, targets)

        indices = out["indices"]
        memories, srcs, masks = list(reversed(memories)), list(reversed(srcs)), list(reversed(masks))
        backbone_feats = list(reversed([feat.tensors for feat in features]))

        bbox_masks, instances_per_batch = self.compute_bbox_att_map(memories, masks, indices, hs)

        features = [memories[1], memories[2], memories[3], backbone_feats[-1]]
        seg_masks = self.mask_head(memories[self.res_lvl], bbox_masks, features, instances_per_batch=instances_per_batch)

        # Compute ouput mask for loss prediction
        out["pred_masks"] = seg_masks

        generate_predictions = targets[0]["generate_predictions"]
        # Compute Inference masks for later use when doing validation
        if not self.training or generate_predictions:
            out["inference_masks"] = self.predict_masks(out, features, srcs, memories, masks, hs)

        return out, targets, None

    # Implements mask computation from forward output for inference taking only into account top k predictions
    def predict_masks(self, out, features, srcs, memories, masks, hs):
        objs_embeddings = self.get_top_k_embeddings(out, hs)

        memories_att_map = [memories[0], memories[2], memories[3]] if self.res_lvl == 0 else [memories[1], memories[2], memories[3]]
        masks_att_map = [masks[0], masks[2], masks[3]] if self.res_lvl == 0 else [masks[1], masks[2], masks[3]]
        bbox_masks = self.bbox_attention(objs_embeddings, memories_att_map, mask=masks_att_map)
        bbox_masks = [bbox_mask.flatten(0, 1) for bbox_mask in bbox_masks]

        out_masks = self.mask_head(memories[self.res_lvl], bbox_masks, features, instances_per_batch=self.top_k_predictions)
        outputs_seg_masks = out_masks.view(hs.shape[1], self.top_k_predictions, out_masks.shape[-2], out_masks.shape[-1])

        return outputs_seg_masks


class DefDETRInstanceSegTopKMutliScaleAttMapsExtraLowResSum(DefDETRInstanceSeg):
    def __init__(self, mask_kwargs, detr_kwargs):
        super().__init__(mask_kwargs, detr_kwargs)

        self.res_lvl = mask_kwargs["attention_map_lvl"]
        hidden_dim, nheads = self.transformer.d_model, self.transformer.nhead

        self.num_levels = 3
        self.bbox_attention = MultiScaleMHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0, num_levels=self.num_levels)
        feats_dims = [256, 256, 256] if self.use_encoded_feats else [1024, 512, 256]
        self.mask_head = MaskHeadSmallConvDefault(hidden_dim + nheads, feats_dims, hidden_dim, nheads=nheads, concat_on_layer=True,
                                                  extra_feat_dim=hidden_dim)

    def compute_bbox_att_map(self, memories, masks, indices, hs):
        memories_att_map = [memories[0], memories[2], memories[3]] if self.res_lvl == 0 else [memories[1], memories[2], memories[3]]
        masks_att_map = [masks[0], masks[2], masks[3]] if self.res_lvl == 0 else [masks[1], masks[2], masks[3]]
        matched_embeddings, instances_per_batch = self.prepare_tmp_batch_fill(indices, hs)
        bbox_masks = self.bbox_attention(matched_embeddings, memories_att_map, mask=masks_att_map)
        indices_to_pick = [torch.arange(0, num_instances) for num_instances in instances_per_batch]
        indices_to_pick = self.get_src_permutation_idx(indices_to_pick)
        bbox_masks = [bbox_mask[indices_to_pick] for bbox_mask in bbox_masks]
        return bbox_masks, instances_per_batch

    def forward(self, samples: NestedTensor, targets: list = None):
        out, targets, features, memories, srcs, masks, hs, pos_embd, inter_references, query_embed = super().forward(samples, targets)

        indices = out["indices"]
        memories, srcs, masks = list(reversed(memories)), list(reversed(srcs)), list(reversed(masks))
        backbone_feats = list(reversed([feat.tensors for feat in features]))

        bbox_masks, instances_per_batch = self.compute_bbox_att_map(memories, masks, indices, hs)

        features = [memories[1], memories[2], memories[3], backbone_feats[-1]] if self.use_encoded_feats else backbone_feats
        seg_masks = self.mask_head(srcs[0], bbox_masks, features, instances_per_batch=instances_per_batch, extra_feat=memories[0])

        # Compute ouput mask for loss prediction
        out["pred_masks"] = seg_masks

        generate_predictions = targets[0]["generate_predictions"]
        # Compute Inference masks for later use when doing validation
        if not self.training or generate_predictions:
            out["inference_masks"] = self.predict_masks(out, features, srcs, memories, masks, hs)

        return out, targets, None

    # Implements mask computation from forward output for inference taking only into account top k predictions
    def predict_masks(self, out, features, srcs, memories, masks, hs):
        objs_embeddings = self.get_top_k_embeddings(out, hs)

        memories_att_map = [memories[0], memories[2], memories[3]] if self.res_lvl == 0 else [memories[1], memories[2], memories[3]]
        masks_att_map = [masks[0], masks[2], masks[3]] if self.res_lvl == 0 else [masks[1], masks[2], masks[3]]
        bbox_masks = self.bbox_attention(objs_embeddings, memories_att_map, mask=masks_att_map)
        bbox_masks = [bbox_mask.flatten(0, 1) for bbox_mask in bbox_masks]

        out_masks = self.mask_head(srcs[0], bbox_masks, features, instances_per_batch=self.top_k_predictions, extra_feat=memories[0])
        outputs_seg_masks = out_masks.view(hs.shape[1], self.top_k_predictions, out_masks.shape[-2], out_masks.shape[-1])

        return outputs_seg_masks


class DefDETRInstanceSegFullUpscale(DefDETRInstanceSeg):
    def __init__(self, mask_kwargs, detr_kwargs):
        super().__init__(mask_kwargs, detr_kwargs)

        hidden_dim, nheads = self.transformer.d_model, self.transformer.nhead

        self.bbox_attention = MHAttentionMapDefault(hidden_dim, hidden_dim, nheads, dropout=0)
        feats_dims = [256, 256, 256, 256] if self.use_encoded_feats else [2048, 1024, 512, 256]
        self.mask_head = MaskHeadSmallConvDefaultFullUpscale(hidden_dim + nheads, feats_dims, hidden_dim)
        self.res_lvl = 0


class DefDETRInstanceSegFullUpscaleMultiScaleAttMapsWithExtraFeat(DefDETRInstanceSeg):
    def __init__(self, mask_kwargs, detr_kwargs):
        super().__init__(mask_kwargs, detr_kwargs)

        hidden_dim, nheads = self.transformer.d_model, self.transformer.nhead

        self.num_levels = 4
        self.bbox_attention = MultiScaleMHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0, num_levels=self.num_levels)
        feats_dims = [256, 256, 256, 256] if self.use_encoded_feats else [2048, 1024, 512, 256]
        self.mask_head = MaskHeadSmallConvDefaultFullUpscale(hidden_dim + nheads, feats_dims, hidden_dim, nheads=nheads, extra_feat_dim=256)

        assert self.batch_mode and not self.fill_batch, "DefDETRInstanceSegFullUpscaleMultiScaleAttMaps only works with batch_mode=True"

    def forward(self, samples: NestedTensor, targets: list = None):
        out, targets, features, memories, srcs, masks, hs, pos_embd, inter_references, query_embed = super().forward(samples, targets)

        indices = out["indices"]
        memories, srcs, masks = list(reversed(memories)), list(reversed(srcs)), list(reversed(masks))
        backbone_feats = list(reversed([feat.tensors for feat in features]))

        matched_embeddings, instances_per_batch = self.prepare_tmp_batch_fill(indices, hs)
        bbox_masks = self.bbox_attention(matched_embeddings, memories, mask=masks)
        indices_to_pick = [torch.arange(0, num_instances) for num_instances in instances_per_batch]
        indices_to_pick = self.get_src_permutation_idx(indices_to_pick)
        bbox_masks = [bbox_mask[indices_to_pick] for bbox_mask in bbox_masks]

        features = [memories[1], memories[2], memories[3], backbone_feats[-1]] if self.use_encoded_feats else backbone_feats
        seg_masks = self.mask_head(srcs[0], bbox_masks, features, instances_per_batch=instances_per_batch, extra_feat=memories[0])

        # Compute ouput mask for loss prediction
        out["pred_masks"] = seg_masks

        generate_predictions = targets[0]["generate_predictions"]
        # Compute Inference masks for later use when doing validation
        if not self.training or generate_predictions:
            out["inference_masks"] = self.predict_masks(out, features, srcs, memories, masks, hs)

        return out, targets, None

    # Implements mask computation from forward output for inference taking only into account top k predictions
    def predict_masks(self, out, features, srcs, memories, masks, hs):
        objs_embeddings = self.get_top_k_embeddings(out, hs)

        bbox_masks = self.bbox_attention(objs_embeddings, memories, mask=masks)
        bbox_masks = [bbox_mask.flatten(0, 1) for bbox_mask in bbox_masks]

        out_masks = self.mask_head(srcs[0], bbox_masks, features, instances_per_batch=self.top_k_predictions, extra_feat=memories[0])
        outputs_seg_masks = out_masks.view(hs.shape[1], self.top_k_predictions, out_masks.shape[-2], out_masks.shape[-1])

        return outputs_seg_masks


class DefDETRInstanceSegMultiScale(DefDETRInstanceSeg):
    def __init__(self, mask_kwargs, detr_kwargs):
        super().__init__(mask_kwargs, detr_kwargs)

        hidden_dim, nheads = self.transformer.d_model, self.transformer.nhead

        self.num_levels = 4
        feats_dims = [256, 256, 256] if self.use_encoded_feats else [1024, 512, 256]
        self.bbox_attention = MultiScaleMHAttentionMap(hidden_dim, hidden_dim, nheads, num_levels=self.num_levels, dropout=0)
        self.mask_head = InstanceSegmSumBatchMaskHead(hidden_dim + nheads, feats_dims, hidden_dim)

    def forward(self, samples: NestedTensor, targets: list = None):

        out, targets, features, memories, srcs, masks, hs, pos_embd, inter_references, query_embed = super().forward(samples, targets)

        indices = out["indices"]
        memories, srcs, masks = list(reversed(memories)), list(reversed(srcs)), list(reversed(masks))
        backbone_feats = list(reversed([feat.tensors for feat in features]))

        if self.fill_batch:
            matched_embeddings, instances_per_batch = self.prepare_batch_fill_random_embeddings(out, hs, indices, targets)
            bbox_masks = self.bbox_attention(matched_embeddings, memories, mask=masks)
            bbox_masks = [bbox_mask.flatten(0, 1) for bbox_mask in bbox_masks]

        elif self.batch_mode:
            matched_embeddings, instances_per_batch = self.prepare_tmp_batch_fill(indices, hs)
            bbox_masks = self.bbox_attention(matched_embeddings, memories, mask=masks)
            indices_to_pick = [torch.arange(0, num_instances) for num_instances in instances_per_batch]
            indices_to_pick = self.get_src_permutation_idx(indices_to_pick)
            bbox_masks = [bbox_mask[indices_to_pick] for bbox_mask in bbox_masks]

        else:
            instances_per_batch = [idx[0].shape[0] for idx in indices]
            bbox_masks = []
            # We have different num of masks to compute for each batch so we need to split computation
            for idx, (embd_idxs, _) in enumerate(indices):
                matched_embeddings = hs[-1, idx, embd_idxs].unsqueeze(0)
                batch_memories = [mem[idx].unsqueeze(0) for mem in memories]
                batch_masks = [mask[idx].unsqueeze(0) for mask in masks]
                bbox_mask = self.bbox_attention(matched_embeddings, batch_memories, mask=batch_masks)
                bbox_masks.append(bbox_mask)

            bbox_masks = [torch.cat([batch_bbox[lvl] for batch_bbox in bbox_masks], dim=1).squeeze(0) for lvl in range(self.num_levels)]

        if self.use_encoded_feats:
            features = [memories[1], memories[2], memories[3], backbone_feats[-1]]
        else:
            features = [srcs[1], srcs[2], srcs[3], backbone_feats[-1]]

        seg_masks = self.mask_head(srcs[0], bbox_masks, features, instances_per_batch=instances_per_batch)

        # Compute ouput mask for loss prediction
        out["pred_masks"] = seg_masks

        generate_predictions = targets[0]["generate_predictions"]
        # Compute Inference masks for later use when doing validation
        if not self.training or generate_predictions:
            out["inference_masks"] = self.predict_masks(out, features, srcs, memories, masks, hs)

        return out, targets, None

    # Implements mask computation from forward output for inference taking only into account top k predictions
    def predict_masks(self, out, features, srcs, memories, masks, hs):
        objs_embeddings = self.get_top_k_embeddings(out, hs)

        bbox_masks = self.bbox_attention(objs_embeddings, memories, mask=masks)
        bbox_masks = [bbox_mask.flatten(0, 1) for bbox_mask in bbox_masks]

        out_masks = self.mask_head(srcs[0], bbox_masks, features, instances_per_batch=self.top_k_predictions)
        outputs_seg_masks = out_masks.view(hs.shape[1], self.top_k_predictions, out_masks.shape[-2], out_masks.shape[-1])

        return outputs_seg_masks


class DefDETRInstSegmDeformableMaskHead(DefDETRInstanceSeg):

    def __init__(self, mask_kwargs, detr_kwargs):
        super().__init__(mask_kwargs, detr_kwargs)

        self.deformable_used_res = mask_kwargs["deformable_used_res"]
        self.mask_head_used_res = mask_kwargs["mask_head_used_res"]
        hidden_dim, nheads = self.transformer.d_model, self.transformer.nhead
        self.def_bbx_attention = MSDeformAttn(d_model=256, n_levels=len(self.deformable_used_res), n_heads=8, n_points=8)
        # self.def_bbx_attention_pytorch = MSDeformAttnPytorch(d_model=256, n_levels=4, n_heads=8, n_points=4)
        self.bbox_attention = MultiScaleMHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0, num_levels=len(self.mask_head_used_res))
        feats_dims = [256, 256, 256] if self.use_encoded_feats else [1024, 512, 256]
        self.mask_head = MaskHeadSmallConvDefault(hidden_dim + nheads, feats_dims, hidden_dim, nheads=nheads, concat_on_layer=False)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def prepare_inputs(self, memories, masks, pos_embeds):
        # prepare input for encoder
        mask_flatten = []
        memories_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (memory, mask, pos_embed) in enumerate(zip(memories, masks, pos_embeds)):
            bs, c, h, w = memory.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            memory = memory.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.transformer.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask_flatten.append(mask)
            memories_flatten.append(memory)
        memories_flatten = torch.cat(memories_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=memories_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.transformer.get_valid_ratio(m) for m in masks], 1)

        return memories_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes, level_start_index, valid_ratios

    def process_reference_points(self, reference_points, valid_ratios):
        if reference_points.shape[-1] == 4:
            reference_points_input = reference_points[:, :, None] \
                                     * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
        else:
            assert reference_points.shape[-1] == 2
            reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

        return reference_points_input

    # Implements mask computation from forward output for inference taking only into account top k predictions
    def predict_masks(self, out, features, srcs, memories, masks, hs, query_embed, reference_points_input, mem_fltn, spatial_shapes,
                      level_start_index, mask_flatten):
        indices = self.get_top_k_indices(out)

        matched_embeddings = torch.gather(hs[-1], 1, indices.unsqueeze(-1).repeat(1, 1, hs[-1].shape[-1]))
        matched_query_embed = torch.gather(query_embed, 1, indices.unsqueeze(-1).repeat(1, 1, query_embed.shape[-1]))
        matched_reference_points_input = torch.gather(reference_points_input, 1,
                                                      indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, reference_points_input.shape[2],
                                                                                                 reference_points_input.shape[3]))

        out_embeddings = self.def_bbx_attention(self.with_pos_embed(matched_embeddings, matched_query_embed),
                                                matched_reference_points_input, mem_fltn, spatial_shapes, level_start_index, mask_flatten)

        memories_att_map = [memories[idx] for idx in self.mask_head_used_res]
        masks_att_map = [masks[idx] for idx in self.mask_head_used_res]

        bbox_masks = self.bbox_attention(out_embeddings, memories_att_map, mask=masks_att_map)
        bbox_masks = [bbox_mask.flatten(0, 1) for bbox_mask in bbox_masks]

        out_masks = self.mask_head(srcs[self.mask_head_used_res[0]], bbox_masks, features, instances_per_batch=self.top_k_predictions)
        outputs_seg_masks = out_masks.view(hs.shape[1], self.top_k_predictions, out_masks.shape[-2], out_masks.shape[-1])
        return outputs_seg_masks

    def forward(self, samples: NestedTensor, targets: list = None):
        out, targets, features, memories, srcs, masks, hs, pos_embd, inter_references, query_embed = super().forward(samples, targets)
        indices = out["indices"]

        used_memories = [memories[-(idx + 1)] for idx in self.deformable_used_res]
        used_masks = [masks[-(idx + 1)] for idx in self.deformable_used_res]
        used_pos_embd = [pos_embd[-(idx + 1)] for idx in self.deformable_used_res]
        mem_fltn, mask_flatten, lvl_pos_embed_fltn, spatial_shapes, level_start_index, valid_ratios = self.prepare_inputs(used_memories,
                                                                                                                          used_masks,
                                                                                                                          used_pos_embd)
        reference_points_input = self.process_reference_points(inter_references[-1], valid_ratios)

        instances_per_batch = [idx[0].shape[0] for idx in indices]
        filled_indices = self.tmp_batch_fill(hs.shape[2], indices)
        num_filled_instances = len(filled_indices[0])
        matched_indices = self.get_src_permutation_idx(filled_indices)
        matched_embeddings = hs[-1][matched_indices].view(hs.shape[1], num_filled_instances, hs.shape[-1])
        matched_query_embed = query_embed[matched_indices].view(hs.shape[1], num_filled_instances, query_embed.shape[-1])
        matched_reference_points_input = reference_points_input[matched_indices].view(hs.shape[1], num_filled_instances,
                                                                                      reference_points_input.shape[2],
                                                                                      reference_points_input.shape[3])

        out_embeddings = self.def_bbx_attention(self.with_pos_embed(matched_embeddings, matched_query_embed),
                                                matched_reference_points_input, mem_fltn, spatial_shapes, level_start_index, mask_flatten)

        memories, srcs, masks = list(reversed(memories)), list(reversed(srcs)), list(reversed(masks))
        backbone_feats = list(reversed([feat.tensors for feat in features]))

        memories_att_map = [memories[idx] for idx in self.mask_head_used_res]
        masks_att_map = [masks[idx] for idx in self.mask_head_used_res]

        # memories_att_map = [memories[0], memories[2], memories[3]] if self.res_lvl == 0 else [memories[1], memories[2], memories[3]]
        # masks_att_map = [masks[0], masks[2], masks[3]] if self.res_lvl == 0 else [masks[1], masks[2], masks[3]]

        bbox_masks = self.bbox_attention(out_embeddings, memories_att_map, mask=masks_att_map)
        indices_to_pick = [torch.arange(0, num_instances) for num_instances in instances_per_batch]
        indices_to_pick = self.get_src_permutation_idx(indices_to_pick)
        bbox_masks = [bbox_mask[indices_to_pick] for bbox_mask in bbox_masks]
        if self.use_encoded_feats:
            features = [memories[idx] for idx in self.mask_head_used_res] + [backbone_feats[-1]]
        else:
            features = [backbone_feats[idx] for idx in self.mask_head_used_res] + [backbone_feats[-1]]

        seg_masks = self.mask_head(srcs[self.mask_head_used_res[0]], bbox_masks, features, instances_per_batch=instances_per_batch)
        # Compute ouput mask for loss prediction
        out["pred_masks"] = seg_masks

        generate_predictions = targets[0]["generate_predictions"]

        # Compute Inference masks for later use when doing validation
        if not self.training or generate_predictions:
            out["inference_masks"] = self.predict_masks(out, features, srcs, memories, masks, hs, query_embed, reference_points_input, mem_fltn,
                                                        spatial_shapes,
                                                        level_start_index, mask_flatten)

        return out, targets, None


class DefDETRInstSegmDeformableSmallFullUpscale(DefDETRInstSegmDeformableMaskHead):

    def __init__(self, mask_kwargs, detr_kwargs):
        super().__init__(mask_kwargs, detr_kwargs)

        self.deformable_used_res = mask_kwargs["deformable_used_res"]
        self.mask_head_used_res = [0, 1, 2, 3]
        hidden_dim, nheads = self.transformer.d_model, self.transformer.nhead
        self.def_bbx_attention = MSDeformAttn(d_model=256, n_levels=len(self.deformable_used_res), n_heads=8, n_points=8)
        self.bbox_attention = MultiScaleMHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0, num_levels=len(self.mask_head_used_res))
        feats_dims = [256, 256, 256] if self.use_encoded_feats else [1024, 512, 256]
        self.mask_head = MaskHeadSmallConvDefaultSmallFullUpscale(hidden_dim + nheads, feats_dims, hidden_dim, nheads=nheads,
                                                                  concat_on_layer=False)


class DefDETRInstSegmDeformableWithMDCnHead(DefDETRInstSegmDeformableMaskHead):

    def __init__(self, mask_kwargs, detr_kwargs):
        super().__init__(mask_kwargs, detr_kwargs)

        hidden_dim, nheads = self.transformer.d_model, self.transformer.nhead
        feats_dims = [256, 256, 256] if self.use_encoded_feats else [1024, 512, 256]
        self.mask_head = MaskHeadDeformableConv(hidden_dim + nheads, feats_dims, hidden_dim, nheads=nheads, concat_on_layer=False)


def build_instance_segm_model(model_name, mask_kwargs, detr_kwargs):
    if model_name == "DefDETRInstanceSegTopK":
        return DefDETRInstanceSegTopK(mask_kwargs, detr_kwargs)
    elif model_name == "DefDETRInstSegmDeformableMaskHead":
        return DefDETRInstSegmDeformableMaskHead(mask_kwargs, detr_kwargs)
    elif model_name == "DefDETRInstanceSegMultiScale":
        return DefDETRInstanceSegMultiScale(mask_kwargs, detr_kwargs)
    elif model_name == "DefDETRInstanceSegFullUpscale":
        return DefDETRInstanceSegFullUpscale(mask_kwargs, detr_kwargs)
    elif model_name == "DefDETRInstanceSegFullUpscaleMultiScaleAttMapsWithExtraFeat":
        return DefDETRInstanceSegFullUpscaleMultiScaleAttMapsWithExtraFeat(mask_kwargs, detr_kwargs)
    elif model_name == "DefDETRInstanceSegTopKMutliScaleAttMaps":
        return DefDETRInstanceSegTopKMutliScaleAttMaps(mask_kwargs, detr_kwargs)
    elif model_name == "DefDETRInstanceSegTopKMutliScaleAttMapAdapterConcat":
        return DefDETRInstanceSegTopKMutliScaleAttMapAdapterConcat(mask_kwargs, detr_kwargs)
    elif model_name == "DefDETRInstanceSegTopKMutliScaleAttMapsOnlyEncodedFeats":
        return DefDETRInstanceSegTopKMutliScaleAttMapsOnlyEncodedFeats(mask_kwargs, detr_kwargs)
    elif model_name == "DefDETRInstanceSegTopKMutliScaleAttMapsExtraLowResSum":
        return DefDETRInstanceSegTopKMutliScaleAttMapsExtraLowResSum(mask_kwargs, detr_kwargs)
    elif model_name == "DefDETRInstSegmDeformableSmallFullUpscale":
        return DefDETRInstSegmDeformableSmallFullUpscale(mask_kwargs, detr_kwargs)
    elif model_name == "DefDETRInstSegmDeformableWithMDCnHead":
        return DefDETRInstSegmDeformableWithMDCnHead(mask_kwargs, detr_kwargs)
    else:
        raise ModuleNotFoundError("Please select valid inst segm model")
