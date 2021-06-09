# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This file provides the definition of the convolutional heads used
to predict masks, as well as the losses.
"""
import io
from collections import defaultdict
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from ..util import box_ops
from ..util.misc import NestedTensor, interpolate

try:
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    pass

from .deformable_detr import DeformableDETR
from .detr import DETR
from .detr_tracking import DETRTrackingBase
from .deformable_segmentation import DETRSegmBase


# class DETRSegmBase(nn.Module):
#     def __init__(self, freeze_detr=False):
#         if freeze_detr:
#             for param in self.parameters():
#                 param.requires_grad_(False)
#
#         nheads = self.transformer.nhead
#         self.bbox_attention = MHAttentionMapDefault(self.hidden_dim, self.hidden_dim, nheads, dropout=0.0)
#
#         self.mask_head = MaskHeadSmallConvDefault(
#             self.hidden_dim + nheads, self.fpn_channels, self.hidden_dim)
#
#     def forward(self, samples: NestedTensor, targets: list = None):
#         out, targets, features, memory, hs, srcs, pos, masks, inter_references, query_embed = super().forward(samples, targets)
#
#         instances_per_batch = hs.shape[2]
#         # Check if it comes from DeTr or Def DeTr
#         if isinstance(memory, list):
#             batch_size = memory[0].shape[0]
#             src, mask, memory = srcs[-3], masks[-3], memory[-3]
#
#             # fpns = [memory[2], memory[1], memory[0]]
#             fpns = [features[-1].tensors, features[-2].tensors, features[-3].tensors, features[-4].tensors]
#
#         else:
#             src, mask = features[-1].decompose()
#             batch_size = src.shape[0]
#             src = self.input_proj(src)
#             fpns = [None, features[2].tensors, features[1].tensors, features[0].tensors]
#
#         # FIXME h_boxes takes the last one computed, keep this in mind
#         bbox_mask = self.bbox_attention(hs[-1], memory, mask=mask)
#         bbox_mask = bbox_mask.flatten(0, 1)
#
#         seg_masks = self.mask_head(src, bbox_mask, fpns, instances_per_batch=instances_per_batch)
#         outputs_seg_masks = seg_masks.view(
#             batch_size, hs.shape[2], seg_masks.shape[-2], seg_masks.shape[-1])
#
#         out["pred_masks"] = outputs_seg_masks
#
#         return out, targets, features, memory, hs, srcs, pos, masks, inter_references, query_embed


# TODO: with meta classes
class DETRSegm(DETRSegmBase, DETR):
    def __init__(self, mask_kwargs, detr_kwargs):
        DETR.__init__(self, **detr_kwargs)
        DETRSegmBase.__init__(self, **mask_kwargs)


class DeformableDETRSegm(DETRSegmBase, DeformableDETR):
    def __init__(self, mask_kwargs, detr_kwargs):
        DeformableDETR.__init__(self, **detr_kwargs)
        DETRSegmBase.__init__(self, **mask_kwargs)


class DETRSegmTracking(DETRSegmBase, DETRTrackingBase, DETR):
    def __init__(self, mask_kwargs, tracking_kwargs, detr_kwargs):
        DETR.__init__(self, **detr_kwargs)
        DETRTrackingBase.__init__(self, **tracking_kwargs)
        DETRSegmBase.__init__(self, **mask_kwargs)


class DeformableDETRSegmTracking(DETRSegmBase, DETRTrackingBase, DeformableDETR):
    def __init__(self, mask_kwargs, tracking_kwargs, detr_kwargs):
        DeformableDETR.__init__(self, **detr_kwargs)
        DETRTrackingBase.__init__(self, **tracking_kwargs)
        DETRSegmBase.__init__(self, **mask_kwargs)


class PostProcessSegm(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes, return_probs=False):
        assert len(orig_target_sizes) == len(max_target_sizes)
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        if "inference_masks" in outputs:
            outputs_masks = outputs["inference_masks"]
        else:
            outputs_masks = outputs["pred_masks"].squeeze(2)
        outputs_masks = F.interpolate(
            outputs_masks,
            size=(max_h, max_w),
            mode="bilinear",
            align_corners=False)

        outputs_masks = outputs_masks.sigmoid().cpu()
        if not return_probs:
            outputs_masks = outputs_masks > self.threshold

        zip_iter = zip(outputs_masks, max_target_sizes, orig_target_sizes)
        for i, (cur_mask, t, tt) in enumerate(zip_iter):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            results[i]["masks"] = F.interpolate(
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            )

            if not return_probs:
                results[i]["masks"] = results[i]["masks"].byte()

        return results


class PostProcessPanoptic(nn.Module):
    """This class converts the output of the model to the final panoptic result,
    in the format expected by the coco panoptic API """

    def __init__(self, is_thing_map, threshold=0.85):
        """
        Parameters:
           is_thing_map: This is a whose keys are the class ids, and the values
                         a boolean indicating whether the class is  a thing (True)
                         or a stuff (False) class
           threshold: confidence threshold: segments with confidence lower than
                      this will be deleted
        """
        super().__init__()
        self.threshold = threshold
        self.is_thing_map = is_thing_map

    def forward(self, outputs, processed_sizes, target_sizes=None):
        """ This function computes the panoptic prediction from the model's predictions.
        Parameters:
            outputs: This is a dict coming directly from the model. See the model
                     doc for the content.
            processed_sizes: This is a list of tuples (or torch tensors) of sizes
                             of the images that were passed to the model, ie the
                             size after data augmentation but before batching.
            target_sizes: This is a list of tuples (or torch tensors) corresponding
                          to the requested final size of each prediction. If left to
                          None, it will default to the processed_sizes
            """
        if target_sizes is None:
            target_sizes = processed_sizes
        assert len(processed_sizes) == len(target_sizes)
        out_logits, raw_masks, raw_boxes = \
            outputs["pred_logits"], outputs["pred_masks"], outputs["pred_boxes"]
        assert len(out_logits) == len(raw_masks) == len(target_sizes)
        preds = []

        def to_tuple(tup):
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.cpu().tolist())

        for cur_logits, cur_masks, cur_boxes, size, target_size in zip(
                out_logits, raw_masks, raw_boxes, processed_sizes, target_sizes
        ):
            # we filter empty queries and detection below threshold
            scores, labels = cur_logits.softmax(-1).max(-1)
            keep = labels.ne(outputs["pred_logits"].shape[-1] - 1) & (scores > self.threshold)
            cur_scores, cur_classes = cur_logits.softmax(-1).max(-1)
            cur_scores = cur_scores[keep]
            cur_classes = cur_classes[keep]
            cur_masks = cur_masks[keep]
            cur_masks = interpolate(cur_masks[None], to_tuple(size), mode="bilinear").squeeze(0)
            cur_boxes = box_ops.box_cxcywh_to_xyxy(cur_boxes[keep])

            h, w = cur_masks.shape[-2:]
            assert len(cur_boxes) == len(cur_classes)

            # It may be that we have several predicted masks for the same stuff class.
            # In the following, we track the list of masks ids for each stuff class
            # (they are merged later on)
            cur_masks = cur_masks.flatten(1)
            stuff_equiv_classes = defaultdict(lambda: [])
            for k, label in enumerate(cur_classes):
                if not self.is_thing_map[label.item()]:
                    stuff_equiv_classes[label.item()].append(k)

            def get_ids_area(masks, scores, dedup=False):
                # This helper function creates the final panoptic segmentation image
                # It also returns the area of the masks that appears on the image

                m_id = masks.transpose(0, 1).softmax(-1)

                if m_id.shape[-1] == 0:
                    # We didn't detect any mask :(
                    m_id = torch.zeros((h, w), dtype=torch.long, device=m_id.device)
                else:
                    m_id = m_id.argmax(-1).view(h, w)

                if dedup:
                    # Merge the masks corresponding to the same stuff class
                    for equiv in stuff_equiv_classes.values():
                        if len(equiv) > 1:
                            for eq_id in equiv:
                                m_id.masked_fill_(m_id.eq(eq_id), equiv[0])

                final_h, final_w = to_tuple(target_size)

                seg_img = Image.fromarray(id2rgb(m_id.view(h, w).cpu().numpy()))
                seg_img = seg_img.resize(size=(final_w, final_h), resample=Image.NEAREST)

                np_seg_img = (torch.ByteTensor(
                    torch.ByteStorage.from_buffer(seg_img.tobytes())).view(final_h, final_w, 3).numpy())
                m_id = torch.from_numpy(rgb2id(np_seg_img))

                area = []
                for i in range(len(scores)):
                    area.append(m_id.eq(i).sum().item())
                return area, seg_img

            area, seg_img = get_ids_area(cur_masks, cur_scores, dedup=True)
            if cur_classes.numel() > 0:
                # We know filter empty masks as long as we find some
                while True:
                    filtered_small = torch.as_tensor([
                        area[i] <= 4
                        for i, c in enumerate(cur_classes)], dtype=torch.bool, device=keep.device)
                    if filtered_small.any().item():
                        cur_scores = cur_scores[~filtered_small]
                        cur_classes = cur_classes[~filtered_small]
                        cur_masks = cur_masks[~filtered_small]
                        area, seg_img = get_ids_area(cur_masks, cur_scores)
                    else:
                        break

            else:
                cur_classes = torch.ones(1, dtype=torch.long, device=cur_classes.device)

            segments_info = []
            for i, a in enumerate(area):
                cat = cur_classes[i].item()
                segments_info.append({
                    "id": i,
                    "isthing": self.is_thing_map[cat],
                    "category_id": cat,
                    "area": a})
            del cur_classes

            with io.BytesIO() as out:
                seg_img.save(out, format="PNG")
                predictions = {"png_string": out.getvalue(), "segments_info": segments_info}
            preds.append(predictions)
        return preds
