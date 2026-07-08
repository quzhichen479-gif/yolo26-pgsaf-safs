"""Crop-view rectification modules for C1-CVRC.

This is a project-agnostic prototype. Codex should adapt it into the actual
YOLO26-C1 trainer after locating the real prediction tensors, assignment
outputs, and C1 specular prior implementation.

The key principle is conservative rectification:
- apply only to tiny/small objects;
- apply only when crop-view mapped boxes are better than full-view boxes;
- keep the original detection loss untouched and add auxiliary losses outside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

Tensor = torch.Tensor


@dataclass
class RectificationConfig:
    lambda_crop: float = 0.35
    lambda_box_rect: float = 0.20
    lambda_quality_rect: float = 0.10
    lambda_spec_neg: float = 0.05
    small_box_area_thr: float = 32.0 * 32.0
    min_crop_iou_gain: float = 0.05
    match_iou_thr: float = 0.30
    rcore_veto_thr: float = 0.65


def box_area_xyxy(boxes: Tensor) -> Tensor:
    wh = (boxes[..., 2:4] - boxes[..., 0:2]).clamp(min=0)
    return wh[..., 0] * wh[..., 1]


def box_iou_xyxy(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = box_area_xyxy(boxes1)[:, None]
    area2 = box_area_xyxy(boxes2)[None, :]
    return inter / (area1 + area2 - inter + eps)


def mean_spec_for_boxes(spec_prior: Optional[Tensor], boxes: Tensor) -> Tuple[Tensor, Tensor]:
    """Return mean Rcore and Eedge per box.

    spec_prior: [2,H,W] or [B,2,H,W]. This helper assumes one image if B exists.
    """
    if boxes.numel() == 0:
        return boxes.new_zeros((0,)), boxes.new_zeros((0,))
    if spec_prior is None or spec_prior.numel() == 0:
        z = boxes.new_zeros((boxes.shape[0],))
        return z, z
    if spec_prior.dim() == 4:
        spec_prior = spec_prior[0]
    if spec_prior.shape[0] < 2:
        z = boxes.new_zeros((boxes.shape[0],))
        return z, z
    _, h, w = spec_prior.shape
    r_vals, e_vals = [], []
    for box in boxes:
        x1, y1, x2, y2 = box.round().long().tolist()
        x1, x2 = max(0, x1), min(w - 1, x2)
        y1, y2 = max(0, y1), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            r_vals.append(box.new_tensor(0.0))
            e_vals.append(box.new_tensor(0.0))
            continue
        patch = spec_prior[:, y1:y2, x1:x2]
        r_vals.append(patch[0].mean())
        e_vals.append(patch[1].mean())
    return torch.stack(r_vals).to(boxes.device), torch.stack(e_vals).to(boxes.device)


class CropViewRectificationLoss(nn.Module):
    """Auxiliary loss for mapping crop-view localization back to full-view C1.

    Expected inputs are post-assignment or decoded candidate tensors. Codex should
    connect this class after locating the real YOLO26 tensors.

    Args:
        full_boxes: [N,4] full-view candidate boxes in original coordinates.
        full_scores: [N] full-view confidence/quality scores.
        crop_boxes_mapped: [M,4] crop-view boxes mapped to original coordinates.
        crop_scores: [M] crop-view confidence/quality scores.
        gt_boxes: [K,4] original-coordinate GT boxes.
        spec_prior: optional [2,H,W] Rcore/Eedge map.
    """

    def __init__(self, cfg: RectificationConfig | None = None):
        super().__init__()
        self.cfg = cfg or RectificationConfig()

    def forward(
        self,
        full_boxes: Tensor,
        full_scores: Tensor,
        crop_boxes_mapped: Tensor,
        crop_scores: Tensor,
        gt_boxes: Tensor,
        spec_prior: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        device = full_boxes.device if full_boxes.numel() else crop_boxes_mapped.device
        zero = torch.zeros((), device=device)
        stats: Dict[str, float] = {
            "num_full": float(full_boxes.shape[0]),
            "num_crop": float(crop_boxes_mapped.shape[0]),
            "num_rectified": 0.0,
            "mean_iou_full": 0.0,
            "mean_iou_crop": 0.0,
            "mean_iou_gain": 0.0,
        }
        if full_boxes.numel() == 0 or crop_boxes_mapped.numel() == 0 or gt_boxes.numel() == 0:
            return zero, stats

        gt_area = box_area_xyxy(gt_boxes)
        small_gt_mask = gt_area <= self.cfg.small_box_area_thr
        small_gt = gt_boxes[small_gt_mask]
        if small_gt.numel() == 0:
            return zero, stats

        full_to_gt_iou = box_iou_xyxy(full_boxes, small_gt)
        crop_to_gt_iou = box_iou_xyxy(crop_boxes_mapped, small_gt)

        full_iou, full_gt_idx = full_to_gt_iou.max(dim=1)
        crop_iou, crop_gt_idx = crop_to_gt_iou.max(dim=1)

        # For every full candidate, find the best crop candidate for the same GT.
        target_boxes = []
        source_boxes = []
        full_q = []
        crop_q = []
        selected_full_iou = []
        selected_crop_iou = []

        rcore_crop, _ = mean_spec_for_boxes(spec_prior, crop_boxes_mapped)

        for i in range(full_boxes.shape[0]):
            if full_iou[i] < self.cfg.match_iou_thr:
                continue
            gt_idx = full_gt_idx[i]
            same_gt = crop_gt_idx == gt_idx
            if not same_gt.any():
                continue
            candidate_ids = torch.where(same_gt)[0]
            local_crop_ious = crop_iou[candidate_ids]
            best_local = torch.argmax(local_crop_ious)
            j = candidate_ids[best_local]
            gain = crop_iou[j] - full_iou[i]
            if gain < self.cfg.min_crop_iou_gain:
                continue
            if rcore_crop[j] > self.cfg.rcore_veto_thr:
                continue
            source_boxes.append(full_boxes[i])
            target_boxes.append(crop_boxes_mapped[j].detach())
            full_q.append(full_scores[i])
            crop_q.append(crop_scores[j].detach())
            selected_full_iou.append(full_iou[i])
            selected_crop_iou.append(crop_iou[j])

        if not target_boxes:
            return zero, stats

        source = torch.stack(source_boxes)
        target = torch.stack(target_boxes)
        full_quality = torch.stack(full_q)
        crop_quality = torch.stack(crop_q)
        full_iou_sel = torch.stack(selected_full_iou)
        crop_iou_sel = torch.stack(selected_crop_iou)

        box_loss = F.smooth_l1_loss(source, target, reduction="mean")
        quality_target = torch.maximum(full_quality.detach(), crop_quality).clamp(0, 1)
        quality_loss = F.smooth_l1_loss(full_quality, quality_target, reduction="mean")

        total = self.cfg.lambda_box_rect * box_loss + self.cfg.lambda_quality_rect * quality_loss
        stats.update(
            {
                "num_rectified": float(source.shape[0]),
                "mean_iou_full": float(full_iou_sel.mean().detach().cpu()),
                "mean_iou_crop": float(crop_iou_sel.mean().detach().cpu()),
                "mean_iou_gain": float((crop_iou_sel - full_iou_sel).mean().detach().cpu()),
                "box_rect_loss": float(box_loss.detach().cpu()),
                "quality_rect_loss": float(quality_loss.detach().cpu()),
            }
        )
        return total, stats


class SpecularHardNegativeLoss(nn.Module):
    """Optional auxiliary penalty for isolated specular-core false positives.

    Penalize confident small predictions that fall in high Rcore / low Eedge regions
    and do not overlap GT. Keep the weight small.
    """

    def __init__(self, cfg: RectificationConfig | None = None):
        super().__init__()
        self.cfg = cfg or RectificationConfig()

    def forward(
        self,
        pred_boxes: Tensor,
        pred_scores: Tensor,
        gt_boxes: Tensor,
        spec_prior: Optional[Tensor],
    ) -> Tuple[Tensor, Dict[str, float]]:
        if pred_boxes.numel() == 0 or spec_prior is None or spec_prior.numel() == 0:
            device = pred_boxes.device if pred_boxes.numel() else pred_scores.device
            return torch.zeros((), device=device), {"num_spec_neg": 0.0}
        iou_to_gt = box_iou_xyxy(pred_boxes, gt_boxes).max(dim=1).values if gt_boxes.numel() else pred_boxes.new_zeros((pred_boxes.shape[0],))
        rcore, eedge = mean_spec_for_boxes(spec_prior, pred_boxes)
        small = box_area_xyxy(pred_boxes) <= self.cfg.small_box_area_thr
        neg = (iou_to_gt < 0.10) & small & (rcore > 0.65) & (eedge < 0.35)
        if not neg.any():
            return pred_scores.new_zeros(()), {"num_spec_neg": 0.0}
        # Penalize confidence only; do not distort boxes.
        loss = F.binary_cross_entropy(pred_scores[neg].clamp(1e-5, 1 - 1e-5), torch.zeros_like(pred_scores[neg]))
        return self.cfg.lambda_spec_neg * loss, {"num_spec_neg": float(neg.sum().detach().cpu()), "spec_neg_loss": float(loss.detach().cpu())}


class ZoomAwareSpatialGate(nn.Module):
    """Residual-safe P3 gate supervised by crop-view regions.

    This is optional and should be enabled only after CVRC loss works. The gate is
    initialized as identity by `gamma = 0`.
    """

    def __init__(self, channels: int, hidden_ratio: float = 0.25):
        super().__init__()
        hidden = max(8, int(channels * hidden_ratio))
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels + 2, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, p3: Tensor, spec_prior_p3: Optional[Tensor] = None, gate_hint: Optional[Tensor] = None) -> Tensor:
        b, _, h, w = p3.shape
        if spec_prior_p3 is None:
            spec_prior_p3 = p3.new_zeros((b, 2, h, w))
        else:
            spec_prior_p3 = F.interpolate(spec_prior_p3, size=(h, w), mode="bilinear", align_corners=False)
            if spec_prior_p3.shape[1] == 1:
                spec_prior_p3 = torch.cat([spec_prior_p3, p3.new_zeros((b, 1, h, w))], dim=1)
        gate = self.gate(torch.cat([p3, spec_prior_p3[:, :2]], dim=1))
        if gate_hint is not None:
            gate_hint = F.interpolate(gate_hint, size=(h, w), mode="nearest")
            gate = torch.maximum(gate, gate_hint.clamp(0, 1))
        return p3 + self.gamma * gate * self.refine(p3)


def make_gate_hint_from_windows(windows_xyxy: Tensor, image_hw: Tuple[int, int], feature_hw: Tuple[int, int], device=None) -> Tensor:
    """Build a binary P3-sized gate hint from selected crop windows."""
    fh, fw = feature_hw
    ih, iw = image_hw
    device = device or windows_xyxy.device
    hint = torch.zeros((1, 1, fh, fw), device=device)
    if windows_xyxy.numel() == 0:
        return hint
    scale_x = fw / max(iw, 1)
    scale_y = fh / max(ih, 1)
    for box in windows_xyxy:
        x1, y1, x2, y2 = box.tolist()
        fx1, fx2 = int(max(0, x1 * scale_x)), int(min(fw, x2 * scale_x + 1))
        fy1, fy2 = int(max(0, y1 * scale_y)), int(min(fh, y2 * scale_y + 1))
        if fx2 > fx1 and fy2 > fy1:
            hint[:, :, fy1:fy2, fx1:fx2] = 1.0
    return hint
