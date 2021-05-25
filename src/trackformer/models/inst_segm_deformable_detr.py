import copy
import random
from ..util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                         get_world_size, interpolate,
                         is_dist_avail_and_initialized, dice_loss, sigmoid_focal_loss)

from .deformable_detr import DeformableDETR
from .inst_segm_modules import *
from ..util import box_ops
from .detr import PostProcess
from .ops.modules import MSDeformAttn, MSDeformAttnPytorch


class DefDETRInstanceSeg(DeformableDETR):
    def __init__(self, mask_kwargs, detr_kwargs):
        super().__init__(**detr_kwargs)

        if mask_kwargs["freeze_detr"]:
            for p in self.parameters():
                p.requires_grad_(False)

        self.top_k_predictions = mask_kwargs["top_k_predictions"]
        self.matcher = mask_kwargs["matcher"]

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


class DefDETRInstanceSegTopK(DefDETRInstanceSeg):
    def __init__(self, mask_kwargs, detr_kwargs):
        super().__init__(mask_kwargs, detr_kwargs)

        hidden_dim, nheads = self.transformer.d_model, self.transformer.nhead

        self.bbox_attention = InstanceSegmDefaultMHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0)
        self.mask_head = InstanceSegmDefaultMaskHead(hidden_dim + nheads, [1024, 512, 256], hidden_dim)
        self.attention_map_lvl = mask_kwargs["attention_map_lvl"]

    def forward(self, samples: NestedTensor, targets: list = None):

        out, targets, features, memories, srcs, masks, hs, pos_embd, inter_references, query_embed = super().forward(samples, targets)

        indices = out["indices"]
        memories, srcs, masks = list(reversed(memories)), list(reversed(srcs)), list(reversed(masks))
        features = list(reversed([feat.tensors for feat in features]))

        if self.fill_batch:
            matched_embeddings, instances_per_batch = self.prepare_batch_fill_random_embeddings(out, hs, indices, targets)
            bbox_masks = self.bbox_attention(matched_embeddings, memories, mask=masks, level=self.attention_map_lvl)
            bbox_masks = bbox_masks.flatten(0, 1)

        elif self.batch_mode:
            matched_embeddings, instances_per_batch = self.prepare_tmp_batch_fill(indices, hs)
            bbox_masks = self.bbox_attention(matched_embeddings, memories, mask=masks, level=self.attention_map_lvl)
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
                bbox_mask = self.bbox_attention(matched_embeddings, batch_memories, mask=batch_masks, level=self.attention_map_lvl)
                bbox_masks.append(bbox_mask)

            bbox_masks = torch.cat(bbox_masks, dim=1).squeeze(0)

        seg_masks = self.mask_head(srcs, bbox_masks, features, instances_per_batch=instances_per_batch, level=self.attention_map_lvl)

        # Compute ouput mask for loss prediction
        out["pred_masks"] = seg_masks

        generate_predictions = targets[0]["generate_predictions"]
        # Compute Inference masks for later use when doing validation
        if not self.training or generate_predictions:
            out["inference_masks"] = self._predict_masks(out, features, srcs, memories, masks, hs)

        return out, targets, None

    # Implements mask computation from forward output for inference taking only into account top k predictions
    def _predict_masks(self, out, features, srcs, memories, masks, hs):
        indices = self.get_top_k_indices(out)
        bs = indices.shape[0]
        objs_embeddings = torch.gather(hs[-1], 1, indices.unsqueeze(-1).repeat(1, 1, hs[-1].shape[-1]))

        bbox_masks = self.bbox_attention(objs_embeddings, memories, mask=masks, level=self.attention_map_lvl)
        bbox_masks = bbox_masks.flatten(0, 1)

        out_masks = self.mask_head(srcs, bbox_masks, features, instances_per_batch=self.top_k_predictions, level=self.attention_map_lvl)
        outputs_seg_masks = out_masks.view(bs, self.top_k_predictions, out_masks.shape[-2], out_masks.shape[-1])

        return outputs_seg_masks


class DefDETRInstSegmDefMaskHead(DefDETRInstanceSeg):

    def __init__(self, mask_kwargs, detr_kwargs):
        super().__init__(mask_kwargs, detr_kwargs)

        hidden_dim, nheads = self.transformer.d_model, self.transformer.nhead
        self.def_bbx_attention = MSDeformAttn(d_model=256, n_levels=4, n_heads=8, n_points=4)
        self.def_bbx_attention_pytorch = MSDeformAttnPytorch(d_model=256, n_levels=4, n_heads=8, n_points=4)
        self.mask_head = InstanceSegmDefaultMaskHead(self.hidden_dim + self.transformer.nhead, [1024, 512, 256], self.hidden_dim)

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

    def forward(self, samples: NestedTensor, targets: list = None):
        out, targets, features, memories, srcs, masks, hs, pos_embd, inter_references, query_embed = super().forward(samples, targets)
        indices = out["indices"]

        mem_fltn, mask_flatten, lvl_pos_embed_fltn, spatial_shapes, level_start_index, valid_ratios = self.prepare_inputs(memories, masks, pos_embd)
        reference_points_input = self.process_reference_points(inter_references[-1], valid_ratios)

        gt_num_instances = [tg["boxes"].shape[0] for tg in targets]
        instances_per_batch = [idx[0].shape[0] for idx in indices]

        # We have different num of masks to compute for each batch so we need to split computation :(
        out["indices"] = indices

        bbox_masks = []
        # We have different num of masks to compute for each batch so we need to split computation :(
        for idx, (embd_idxs, _) in enumerate(indices):
            matched_embeddings = hs[-1, idx, embd_idxs].unsqueeze(0)
            matched_query_embed = query_embed[idx, embd_idxs].unsqueeze(0)
            matched_reference_points_input = reference_points_input[idx, embd_idxs].unsqueeze(0)
            batch_memory = mem_fltn[idx].unsqueeze(0)
            batch_mask = mask_flatten[idx].unsqueeze(0)

            out_masks = self.def_bbx_attention(self.with_pos_embed(matched_embeddings, matched_query_embed),
                                               matched_reference_points_input, batch_memory, spatial_shapes, level_start_index, batch_mask)
            out_masks_pytorch = self.def_bbx_attention_pytorch(self.with_pos_embed(matched_embeddings, matched_query_embed),
                                               matched_reference_points_input, batch_memory, spatial_shapes, level_start_index, batch_mask)
            bbox_masks.append(out_masks)

        print("a")


def build_instance_segm_model(model_name, mask_kwargs, detr_kwargs):
    if model_name == "DefDETRInstanceSegTopK":
        return DefDETRInstanceSegTopK(mask_kwargs, detr_kwargs)
    elif model_name == "DefDETRInstSegmDefMaskHead":
        return DefDETRInstSegmDefMaskHead(mask_kwargs, detr_kwargs)
    else:
        raise ModuleNotFoundError("Please select valid inst segm model")


class InstSegmBoxPostProcess(PostProcess):
    def __init__(self, top_k_predictions=100):
        super().__init__()
        self.top_k_predictions = top_k_predictions

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        top_k_indexes = outputs["top_k_indexes"]
        scores = outputs["top_k_values"]

        topk_boxes = top_k_indexes // out_logits.shape[2]
        labels = top_k_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'scores_no_object': 1 - s, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results


class InstSegmMaskPostProcess(nn.Module):
    def __init__(self, top_k_predictions=100, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.top_k_predictions = top_k_predictions

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        assert len(orig_target_sizes) == len(max_target_sizes)
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        outputs_masks = outputs["inference_masks"]

        outputs_masks = F.interpolate(outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False)
        outputs_masks = (outputs_masks.sigmoid() > self.threshold).cpu()

        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            results[i]["masks"] = F.interpolate(
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            ).byte()

        return results
