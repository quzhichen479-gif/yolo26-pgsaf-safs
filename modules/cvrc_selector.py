"""CVRC crop selector prototype.

This file is a Codex-ready skeleton for integration into the actual YOLO26-C1
repository. It intentionally avoids project-specific imports so it can be copied
into `ultralytics/nn/modules/` or `ultralytics/utils/` and adapted to the real
trainer/dataloader interfaces.

Coordinate convention:
- boxes are absolute xyxy in the original image coordinate system.
- image size is (height, width).
- labels may be converted from normalized xywh by `labels_xywhn_to_xyxy`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


@dataclass
class CVRCCropConfig:
    crop_imgsz: int = 640
    context_ratio: float = 0.05
    max_positive_crops_per_image: int = 3
    max_total_crops_per_image: int = 4
    hard_negative_ratio: float = 0.10
    min_box_side_px: float = 2.0
    small_box_area_thr: float = 32.0 * 32.0
    low_conf_thr: float = 0.25
    high_conf_thr: float = 0.55
    nms_iou_thr: float = 0.70
    min_crop_side_px: float = 96.0


@dataclass
class CropWindow:
    xyxy: Tensor
    score: float
    source: str
    meta: dict


def clip_xyxy(boxes: Tensor, image_hw: Tuple[int, int]) -> Tensor:
    """Clip absolute xyxy boxes to image boundaries."""
    h, w = image_hw
    out = boxes.clone()
    out[..., 0::2].clamp_(0, w - 1)
    out[..., 1::2].clamp_(0, h - 1)
    return out


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


def labels_xywhn_to_xyxy(labels_xywhn: Tensor, image_hw: Tuple[int, int]) -> Tensor:
    """Convert normalized xywh labels to absolute xyxy.

    Accepts either Nx4 or Nx5/6 tensors. If class columns exist, pass only the
    xywh part before calling this helper.
    """
    h, w = image_hw
    xywh = labels_xywhn.float()
    x, y, bw, bh = xywh.unbind(-1)
    x1 = (x - bw / 2.0) * w
    y1 = (y - bh / 2.0) * h
    x2 = (x + bw / 2.0) * w
    y2 = (y + bh / 2.0) * h
    return clip_xyxy(torch.stack([x1, y1, x2, y2], dim=-1), image_hw)


def expand_box_to_crop(
    box: Tensor,
    image_hw: Tuple[int, int],
    context_ratio: float = 0.05,
    min_crop_side_px: float = 96.0,
    square: bool = True,
) -> Tensor:
    """Expand a target box into a crop window with context.

    The crop is optionally squared so resize-to-640 does not distort too much.
    """
    h, w = image_hw
    x1, y1, x2, y2 = box.float()
    bw = (x2 - x1).clamp(min=1.0)
    bh = (y2 - y1).clamp(min=1.0)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    side_w = bw * (1.0 + 2.0 * context_ratio)
    side_h = bh * (1.0 + 2.0 * context_ratio)
    if square:
        side = torch.maximum(torch.maximum(side_w, side_h), box.new_tensor(min_crop_side_px))
        side_w = side_h = side
    else:
        side_w = torch.maximum(side_w, box.new_tensor(min_crop_side_px))
        side_h = torch.maximum(side_h, box.new_tensor(min_crop_side_px))

    nx1 = cx - side_w / 2.0
    ny1 = cy - side_h / 2.0
    nx2 = cx + side_w / 2.0
    ny2 = cy + side_h / 2.0

    # Shift back inside the image while preserving crop size when possible.
    dx1 = torch.clamp(-nx1, min=0)
    dy1 = torch.clamp(-ny1, min=0)
    nx1, nx2 = nx1 + dx1, nx2 + dx1
    ny1, ny2 = ny1 + dy1, ny2 + dy1
    dx2 = torch.clamp(nx2 - (w - 1), min=0)
    dy2 = torch.clamp(ny2 - (h - 1), min=0)
    nx1, nx2 = nx1 - dx2, nx2 - dx2
    ny1, ny2 = ny1 - dy2, ny2 - dy2

    return clip_xyxy(torch.stack([nx1, ny1, nx2, ny2]), image_hw)


def greedy_window_nms(windows: Sequence[CropWindow], iou_thr: float) -> List[CropWindow]:
    if not windows:
        return []
    ordered = sorted(windows, key=lambda x: x.score, reverse=True)
    kept: List[CropWindow] = []
    for win in ordered:
        if not kept:
            kept.append(win)
            continue
        boxes_kept = torch.stack([k.xyxy for k in kept]).to(win.xyxy.device)
        ious = box_iou_xyxy(win.xyxy[None, :], boxes_kept)[0]
        if float(ious.max()) < iou_thr:
            kept.append(win)
    return kept


def mean_spec_in_box(spec_prior: Optional[Tensor], box: Tensor) -> Tuple[float, float]:
    """Return mean Rcore/Eedge in a crop box.

    spec_prior shape can be [2,H,W] or [1/2,H,W]. Missing values return 0.
    """
    if spec_prior is None or spec_prior.numel() == 0:
        return 0.0, 0.0
    if spec_prior.dim() == 4:
        spec_prior = spec_prior[0]
    c, h, w = spec_prior.shape
    x1, y1, x2, y2 = box.round().long().tolist()
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0
    patch = spec_prior[:, y1:y2, x1:x2]
    rcore = float(patch[0].mean()) if c >= 1 else 0.0
    eedge = float(patch[1].mean()) if c >= 2 else 0.0
    return rcore, eedge


def build_gt_positive_crops(
    gt_xyxy: Tensor,
    image_hw: Tuple[int, int],
    cfg: CVRCCropConfig,
    spec_prior: Optional[Tensor] = None,
) -> List[CropWindow]:
    """Build warmup positive crops from tiny/small GT boxes."""
    if gt_xyxy.numel() == 0:
        return []
    wh = (gt_xyxy[:, 2:4] - gt_xyxy[:, 0:2]).clamp(min=0)
    area = wh[:, 0] * wh[:, 1]
    min_side = wh.min(dim=1).values
    keep = (area <= cfg.small_box_area_thr) & (min_side >= cfg.min_box_side_px)
    small_boxes = gt_xyxy[keep]
    if small_boxes.numel() == 0:
        return []

    # Prefer smaller boxes because they benefit most from crop-view resize.
    order = torch.argsort(box_area_xyxy(small_boxes))
    windows: List[CropWindow] = []
    for idx in order[: cfg.max_positive_crops_per_image].tolist():
        crop = expand_box_to_crop(
            small_boxes[idx],
            image_hw=image_hw,
            context_ratio=cfg.context_ratio,
            min_crop_side_px=cfg.min_crop_side_px,
            square=True,
        )
        rcore, eedge = mean_spec_in_box(spec_prior, crop)
        score = 1.0 + 0.8 * eedge - 1.2 * rcore
        windows.append(CropWindow(crop, float(score), "gt_positive", {"rcore": rcore, "eedge": eedge}))
    return greedy_window_nms(windows, cfg.nms_iou_thr)


def build_prediction_crops(
    pred_xyxy: Tensor,
    pred_scores: Tensor,
    image_hw: Tuple[int, int],
    cfg: CVRCCropConfig,
    spec_prior: Optional[Tensor] = None,
) -> List[CropWindow]:
    """Build selective crops from full-view low-confidence small predictions."""
    if pred_xyxy.numel() == 0:
        return []
    wh = (pred_xyxy[:, 2:4] - pred_xyxy[:, 0:2]).clamp(min=0)
    area = wh[:, 0] * wh[:, 1]
    is_small = area <= cfg.small_box_area_thr
    is_low_conf = (pred_scores >= cfg.low_conf_thr) & (pred_scores <= cfg.high_conf_thr)
    keep = is_small & is_low_conf
    boxes = pred_xyxy[keep]
    scores = pred_scores[keep]
    if boxes.numel() == 0:
        return []

    windows: List[CropWindow] = []
    for box, conf in zip(boxes, scores):
        crop = expand_box_to_crop(
            box,
            image_hw=image_hw,
            context_ratio=cfg.context_ratio,
            min_crop_side_px=cfg.min_crop_side_px,
            square=True,
        )
        rcore, eedge = mean_spec_in_box(spec_prior, crop)
        uncertainty = 1.0 - abs(float(conf) - 0.5) * 2.0
        score = 1.0 * float(conf) + 0.8 * eedge - 1.2 * rcore + 0.5 * uncertainty
        windows.append(CropWindow(crop, float(score), "pred_lowconf_small", {"conf": float(conf), "rcore": rcore, "eedge": eedge}))
    return greedy_window_nms(windows, cfg.nms_iou_thr)


def build_edge_prior_crops(
    spec_prior: Optional[Tensor],
    image_hw: Tuple[int, int],
    cfg: CVRCCropConfig,
    topk: int = 2,
) -> List[CropWindow]:
    """Build a few reflectance-edge crops from high Eedge / low Rcore locations.

    This is deliberately sparse. It is not ordinary SAHI.
    """
    if spec_prior is None or spec_prior.numel() == 0:
        return []
    if spec_prior.dim() == 4:
        spec_prior = spec_prior[0]
    if spec_prior.shape[0] < 2:
        return []
    rcore, eedge = spec_prior[0:1], spec_prior[1:2]
    score_map = (eedge - 0.75 * rcore).clamp(min=0.0)
    pooled = F.max_pool2d(score_map[None], kernel_size=31, stride=16, padding=15)[0, 0]
    values, flat_idx = torch.topk(pooled.flatten(), k=min(topk, pooled.numel()))
    h, w = image_hw
    windows: List[CropWindow] = []
    for value, idx in zip(values, flat_idx):
        if float(value) <= 0:
            continue
        py = int(idx // pooled.shape[1])
        px = int(idx % pooled.shape[1])
        cx = torch.tensor(float(px * 16), device=spec_prior.device)
        cy = torch.tensor(float(py * 16), device=spec_prior.device)
        half = torch.tensor(float(cfg.min_crop_side_px), device=spec_prior.device)
        raw = torch.stack([cx - half, cy - half, cx + half, cy + half])
        crop = clip_xyxy(raw, image_hw)
        rr, ee = mean_spec_in_box(spec_prior, crop)
        score = 0.8 * ee - 1.2 * rr
        windows.append(CropWindow(crop, float(score), "edge_prior", {"rcore": rr, "eedge": ee}))
    return greedy_window_nms(windows, cfg.nms_iou_thr)


def select_cvrc_crops(
    image_hw: Tuple[int, int],
    cfg: CVRCCropConfig,
    gt_xyxy: Optional[Tensor] = None,
    pred_xyxy: Optional[Tensor] = None,
    pred_scores: Optional[Tensor] = None,
    spec_prior: Optional[Tensor] = None,
    warmup: bool = True,
) -> List[CropWindow]:
    """Select CVRC crop windows for one image.

    Integration note:
    - During warmup, pass GT boxes and set `warmup=True`.
    - After warmup, pass both GT and full-branch predictions.
    - Always cap by `max_total_crops_per_image`.
    """
    windows: List[CropWindow] = []
    if gt_xyxy is not None:
        windows.extend(build_gt_positive_crops(gt_xyxy, image_hw, cfg, spec_prior))
    if not warmup and pred_xyxy is not None and pred_scores is not None:
        windows.extend(build_prediction_crops(pred_xyxy, pred_scores, image_hw, cfg, spec_prior))
        remaining = max(0, cfg.max_total_crops_per_image - len(windows))
        if remaining > 0:
            windows.extend(build_edge_prior_crops(spec_prior, image_hw, cfg, topk=remaining))

    windows = greedy_window_nms(windows, cfg.nms_iou_thr)
    return sorted(windows, key=lambda x: x.score, reverse=True)[: cfg.max_total_crops_per_image]


def map_boxes_from_crop_to_original(boxes_xyxy: Tensor, crop_xyxy: Tensor, crop_imgsz: int) -> Tensor:
    """Map crop-resized xyxy predictions back to original image coordinates."""
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy
    x1, y1, x2, y2 = crop_xyxy.float()
    crop_w = (x2 - x1).clamp(min=1.0)
    crop_h = (y2 - y1).clamp(min=1.0)
    scale = boxes_xyxy.new_tensor([crop_w / crop_imgsz, crop_h / crop_imgsz, crop_w / crop_imgsz, crop_h / crop_imgsz])
    offset = boxes_xyxy.new_tensor([x1, y1, x1, y1])
    return boxes_xyxy * scale + offset


def map_boxes_from_original_to_crop(boxes_xyxy: Tensor, crop_xyxy: Tensor, crop_imgsz: int) -> Tensor:
    """Map original-image xyxy boxes into the resized crop coordinate system."""
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy
    x1, y1, x2, y2 = crop_xyxy.float()
    crop_w = (x2 - x1).clamp(min=1.0)
    crop_h = (y2 - y1).clamp(min=1.0)
    offset = boxes_xyxy.new_tensor([x1, y1, x1, y1])
    scale = boxes_xyxy.new_tensor([crop_imgsz / crop_w, crop_imgsz / crop_h, crop_imgsz / crop_w, crop_imgsz / crop_h])
    return (boxes_xyxy - offset) * scale
