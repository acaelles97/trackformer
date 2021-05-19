import copy
from ..util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                         get_world_size, interpolate,
                         is_dist_avail_and_initialized, dice_loss, sigmoid_focal_loss)

from .deformable_detr import DeformableDETR
from .inst_segm_modules import *
from ..util import box_ops
from .detr import PostProcess


class DefDETRInstanceSegTopK(DeformableDETR):
    def __init__(self, mask_kwargs, detr_kwargs):
        super().__init__(**detr_kwargs)

        if mask_kwargs["freeze_detr"]:
            for p in self.parameters():
                p.requires_grad_(False)

        self.top_k_predictions = mask_kwargs["top_k_predictions"]
        self.matcher = mask_kwargs["matcher"]

        hidden_dim, nheads = self.transformer.d_model, self.transformer.nhead

        self.bbox_attention = InstanceSegmDefaultMHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0)
        self.mask_head = InstanceSegmDefaultMaskHead(hidden_dim + nheads, [1024, 512, 256], hidden_dim)
        self.data = {}
        self.encode_references = mask_kwargs["encode_references"]
        self.attention_map_lvl = mask_kwargs["attention_map_lvl"]

    def get_top_k_indices(self, outputs):
        out_logits = outputs['pred_logits']
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.top_k_predictions, dim=1)
        outputs["top_k_values"] = topk_values
        outputs["top_k_indexes"] = topk_indexes
        return topk_indexes // out_logits.shape[2]

    def forward(self, samples: NestedTensor, targets: list = None):
        out, targets, features, memories, hs, srcs, masks, inter_references = super().forward(samples, targets)

        memories = list(reversed(memories))
        srcs = list(reversed(srcs))
        masks = list(reversed(masks))
        features = list(reversed([feat.tensors for feat in features]))

        outputs_without_aux = {k: v for k, v in out.items() if k != 'aux_outputs' and k != 'enc_outputs' and k != 'hs_embed'}
        indices = self.matcher(outputs_without_aux, targets)

        gt_num_instances = [tg["boxes"].shape[0] for tg in targets]
        instances_per_batch = [idx[0].shape[0] for idx in indices]

        assert instances_per_batch == gt_num_instances
        gt_num_instances = sum(gt_num_instances)

        # We have different num of masks to compute for each batch so we need to split computation :(
        out["indices"] = indices

        bbox_masks = []
        # We have different num of masks to compute for each batch so we need to split computation :(
        for idx, (embd_idxs, _) in enumerate(indices):
            matched_embeddings = hs[-1, idx, embd_idxs].unsqueeze(0)

            if self.encode_references:
                matched_references = inter_references[-1, idx, embd_idxs].unsqueeze(0)
                matched_embeddings = torch.cat([matched_embeddings, matched_references], dim=2)

            batch_memories = [mem[idx].unsqueeze(0) for mem in memories]
            batch_masks = [mask[idx].unsqueeze(0) for mask in masks]
            bbox_mask = self.bbox_attention(matched_embeddings, batch_memories, mask=batch_masks, level=self.attention_map_lvl)
            bbox_masks.append(bbox_mask)

        seg_masks = self.mask_head(srcs, bbox_masks, features, instances_per_batch=instances_per_batch, level=self.attention_map_lvl)
        assert gt_num_instances == seg_masks.shape[0]

        # Compute ouput mask for loss prediction
        seg_masks = seg_masks.squeeze(1)
        out["pred_masks"] = seg_masks

        # Compute Inference masks for later use when doing validation
        if not self.training:
            out["inference_masks"] = self._predict_masks(out,  features, srcs, memories, masks,  hs, inter_references)

        return out, targets, None

    # Implements mask computation from forward output for inference taking only into account top k predictions
    def _predict_masks(self, out,  features, srcs, memories, masks,  hs, inter_references):
        indices = self.get_top_k_indices(out)
        bs = indices.shape[0]
        objs_embeddings = torch.gather(hs[-1], 1, indices.unsqueeze(-1).repeat(1, 1, hs[-1].shape[-1]))

        if self.encode_references:
            matched_references = torch.gather(inter_references[-1], 1,
                                              indices.unsqueeze(-1).repeat(1, 1, inter_references[-1].shape[-1]))
            objs_embeddings = torch.cat([objs_embeddings, matched_references], dim=2)

        bbox_masks = self.bbox_attention(objs_embeddings, memories, mask=masks, level=self.attention_map_lvl)
        bbox_masks = bbox_masks.flatten(0, 1)
        # On test time batch is fixes
        instances_per_batch = [self.top_k_predictions for _ in range(bs)]

        out_masks = self.mask_head(srcs, [bbox_masks], features, instances_per_batch=instances_per_batch,
                                   level=self.attention_map_lvl)
        outputs_seg_masks = out_masks.view(bs, self.top_k_predictions, out_masks.shape[-2], out_masks.shape[-1])

        return outputs_seg_masks


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

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
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
