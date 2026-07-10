"""Training-only Zoom-View Correction helpers for YOLO26-C1.

Reference code only. The actual integration must keep the C1 inference graph,
Detect head, assignment, and base detection loss unchanged. The crop teacher
must be frozen and used only in training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class CropMeta:
    image_index: int
    crop_xyxy_full: Tuple[float, float, float, float]
    crop_hw: Tuple[int, int]
    full_hw: Tuple[int, int]


@dataclass
class ZVCConfig:
    temperature: float = 2.0
    cls_weight: float = 0.25
    box_weight: float = 0.25
    hardneg_weight: float = 0.05
    teacher_pos_conf: float = 0.35
    teacher_pos_iou: float = 0.50
    teacher_neg_conf_max: float = 0.05
    student_hardneg_conf: float = 0.10
    match_iou_min: float = 0.10
    box_loss_type: str = "smooth_l1"
    warmup_epochs: int = 10
    ramp_epochs: int = 20
    max_positive_pairs_per_image: int = 64
    max_hardneg_candidates_per_crop: int = 32
    eps: float = 1e-7


def _check_boxes(boxes: Tensor, name: str) -> None:
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"{name} must be [N,4], got {tuple(boxes.shape)}")


def box_area_xyxy(boxes: Tensor) -> Tensor:
    _check_boxes(boxes, "boxes")
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
    return wh[:, 0] * wh[:, 1]


def pairwise_iou_xyxy(a: Tensor, b: Tensor, eps: float = 1e-7) -> Tensor:
    _check_boxes(a, "a")
    _check_boxes(b, "b")
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.shape[0], b.shape[0]))
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = box_area_xyxy(a)[:, None] + box_area_xyxy(b)[None, :] - inter
    return inter / (union + eps)


def aligned_giou_loss_xyxy(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    _check_boxes(pred, "pred")
    _check_boxes(target, "target")
    if pred.shape != target.shape:
        raise ValueError("pred and target must have identical shapes")
    if pred.numel() == 0:
        return pred.new_zeros((0,))
    lt = torch.maximum(pred[:, :2], target[:, :2])
    rb = torch.minimum(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = box_area_xyxy(pred) + box_area_xyxy(target) - inter
    iou = inter / (union + eps)
    enc_lt = torch.minimum(pred[:, :2], target[:, :2])
    enc_rb = torch.maximum(pred[:, 2:], target[:, 2:])
    enc_wh = (enc_rb - enc_lt).clamp(min=0)
    enc_area = enc_wh[:, 0] * enc_wh[:, 1]
    return 1.0 - (iou - (enc_area - union) / (enc_area + eps))


def _crop(meta: CropMeta, ref: Tensor) -> Tensor:
    crop = ref.new_tensor(meta.crop_xyxy_full)
    if crop.numel() != 4 or crop[2] <= crop[0] or crop[3] <= crop[1]:
        raise ValueError(f"invalid crop: {meta.crop_xyxy_full}")
    return crop


def map_boxes_full_to_crop(boxes: Tensor, meta: CropMeta, clip: bool = True) -> Tensor:
    _check_boxes(boxes, "boxes")
    c = _crop(meta, boxes)
    out_h, out_w = meta.crop_hw
    sx, sy = out_w / (c[2] - c[0]), out_h / (c[3] - c[1])
    out = boxes.clone()
    out[:, [0, 2]] = (out[:, [0, 2]] - c[0]) * sx
    out[:, [1, 3]] = (out[:, [1, 3]] - c[1]) * sy
    if clip:
        out[:, [0, 2]] = out[:, [0, 2]].clamp(0, float(out_w))
        out[:, [1, 3]] = out[:, [1, 3]].clamp(0, float(out_h))
    return out


def map_boxes_crop_to_full(boxes: Tensor, meta: CropMeta, clip: bool = True) -> Tensor:
    _check_boxes(boxes, "boxes")
    c = _crop(meta, boxes)
    out_h, out_w = meta.crop_hw
    sx, sy = (c[2] - c[0]) / out_w, (c[3] - c[1]) / out_h
    out = boxes.clone()
    out[:, [0, 2]] = out[:, [0, 2]] * sx + c[0]
    out[:, [1, 3]] = out[:, [1, 3]] * sy + c[1]
    if clip:
        full_h, full_w = meta.full_hw
        out[:, [0, 2]] = out[:, [0, 2]].clamp(0, float(full_w))
        out[:, [1, 3]] = out[:, [1, 3]].clamp(0, float(full_h))
    return out


def make_square_crop_around_box(
    box: Tensor,
    full_hw: Tuple[int, int],
    *,
    context_scale: float = 2.5,
    min_crop_size: float = 96.0,
    max_crop_size: Optional[float] = None,
    center_jitter: float = 0.10,
    scale_jitter: float = 0.15,
    generator: Optional[torch.Generator] = None,
) -> Tuple[float, float, float, float]:
    if box.ndim != 1 or box.numel() != 4:
        raise ValueError("box must be [4]")
    full_h, full_w = full_hw
    if full_h <= 0 or full_w <= 0:
        raise ValueError("full_hw must be positive")
    x1, y1, x2, y2 = [float(v) for v in box.detach().cpu()]
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    limit = float(min(full_h, full_w) if max_crop_size is None else max_crop_size)

    def signed_rand() -> float:
        return 2.0 * float(torch.rand((), generator=generator)) - 1.0

    size = max(max(w, h) * context_scale, min_crop_size)
    size = min(max(size * (1 + scale_jitter * signed_rand()), min_crop_size), limit)
    cx += center_jitter * size * signed_rand()
    cy += center_jitter * size * signed_rand()
    left = min(max(cx - size / 2, 0.0), max(float(full_w) - size, 0.0))
    top = min(max(cy - size / 2, 0.0), max(float(full_h) - size, 0.0))
    return left, top, min(left + size, float(full_w)), min(top + size, float(full_h))


def visible_fraction_in_crop(boxes: Tensor, meta: CropMeta, eps: float = 1e-7) -> Tensor:
    _check_boxes(boxes, "boxes")
    c = _crop(meta, boxes)
    lt = torch.maximum(boxes[:, :2], c[:2])
    rb = torch.minimum(boxes[:, 2:], c[2:])
    wh = (rb - lt).clamp(min=0)
    return (wh[:, 0] * wh[:, 1]) / box_area_xyxy(boxes).clamp(min=eps)


def centers_inside_crop(boxes: Tensor, meta: CropMeta) -> Tensor:
    _check_boxes(boxes, "boxes")
    c = _crop(meta, boxes)
    ctr = (boxes[:, :2] + boxes[:, 2:]) / 2
    return (
        (ctr[:, 0] >= c[0]) & (ctr[:, 0] <= c[2])
        & (ctr[:, 1] >= c[1]) & (ctr[:, 1] <= c[3])
    )


def ramp_weight(epoch: int, cfg: ZVCConfig) -> float:
    if epoch < cfg.warmup_epochs:
        return 0.0
    if cfg.ramp_epochs <= 0:
        return 1.0
    return min(1.0, (epoch - cfg.warmup_epochs + 1) / float(cfg.ramp_epochs))


@torch.no_grad()
def select_reliable_teacher_positives(
    teacher_boxes: Tensor,
    teacher_logits: Tensor,
    gt_boxes: Tensor,
    gt_labels: Tensor,
    cfg: ZVCConfig,
) -> Dict[str, Tensor]:
    _check_boxes(teacher_boxes, "teacher_boxes")
    _check_boxes(gt_boxes, "gt_boxes")
    if teacher_logits.ndim != 2 or gt_labels.ndim != 1:
        raise ValueError("teacher_logits must be [N,C] and gt_labels [G]")
    device, dtype = teacher_boxes.device, teacher_boxes.dtype
    empty_l = torch.empty(0, dtype=torch.long, device=device)
    empty_f = torch.empty(0, dtype=dtype, device=device)
    empty = {
        "teacher_indices": empty_l, "gt_indices": empty_l, "labels": empty_l,
        "confidence": empty_f, "iou": empty_f, "quality": empty_f,
    }
    if teacher_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return empty

    probs = teacher_logits.sigmoid()
    ious = pairwise_iou_xyxy(teacher_boxes, gt_boxes, cfg.eps)
    records = []
    for gi in range(gt_boxes.shape[0]):
        label = int(gt_labels[gi])
        if not 0 <= label < probs.shape[1]:
            continue
        score = probs[:, label]
        ti = int(torch.argmax(score * ious[:, gi]))
        conf, iou = score[ti], ious[ti, gi]
        if conf >= cfg.teacher_pos_conf and iou >= cfg.teacher_pos_iou:
            records.append((ti, gi, label, conf, iou, torch.sqrt((conf * iou).clamp(0, 1))))
    if not records:
        return empty

    best = {}
    for rec in records:
        if rec[0] not in best or rec[-1] > best[rec[0]][-1]:
            best[rec[0]] = rec
    records = sorted(best.values(), key=lambda r: float(r[-1]), reverse=True)
    return {
        "teacher_indices": torch.tensor([r[0] for r in records], device=device),
        "gt_indices": torch.tensor([r[1] for r in records], device=device),
        "labels": torch.tensor([r[2] for r in records], device=device),
        "confidence": torch.stack([r[3] for r in records]),
        "iou": torch.stack([r[4] for r in records]),
        "quality": torch.stack([r[5] for r in records]),
    }


@torch.no_grad()
def match_student_to_teacher(
    student_boxes: Tensor,
    teacher_boxes: Tensor,
    quality: Tensor,
    cfg: ZVCConfig,
) -> Dict[str, Tensor]:
    _check_boxes(student_boxes, "student_boxes")
    _check_boxes(teacher_boxes, "teacher_boxes")
    empty = torch.empty(0, dtype=torch.long, device=student_boxes.device)
    if student_boxes.numel() == 0 or teacher_boxes.numel() == 0:
        return {"student_indices": empty, "teacher_indices": empty}

    ious = pairwise_iou_xyxy(student_boxes.detach(), teacher_boxes.detach(), cfg.eps)
    used = torch.zeros(student_boxes.shape[0], dtype=torch.bool, device=student_boxes.device)
    s_idx, t_idx = [], []
    for ti in torch.argsort(quality.detach(), descending=True).tolist():
        row = ious[:, ti].clone()
        row[used] = -1
        si = int(torch.argmax(row))
        if row[si] < cfg.match_iou_min:
            continue
        used[si] = True
        s_idx.append(si)
        t_idx.append(ti)
        if len(s_idx) >= cfg.max_positive_pairs_per_image:
            break
    return {
        "student_indices": torch.tensor(s_idx, dtype=torch.long, device=student_boxes.device),
        "teacher_indices": torch.tensor(t_idx, dtype=torch.long, device=student_boxes.device),
    }


def positive_zoom_distillation_loss(
    student_boxes_full: Tensor,
    student_logits: Tensor,
    teacher_boxes_crop: Tensor,
    teacher_logits: Tensor,
    gt_boxes_crop: Tensor,
    gt_labels: Tensor,
    meta: CropMeta,
    cfg: ZVCConfig,
    epoch: int,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    _check_boxes(student_boxes_full, "student_boxes_full")
    zero = student_logits.sum() * 0.0
    selected = select_reliable_teacher_positives(
        teacher_boxes_crop, teacher_logits, gt_boxes_crop, gt_labels, cfg
    )
    if selected["teacher_indices"].numel() == 0:
        return zero, _positive_logs(zero)

    ti = selected["teacher_indices"]
    teacher_full = map_boxes_crop_to_full(teacher_boxes_crop[ti], meta)
    matched = match_student_to_teacher(student_boxes_full, teacher_full, selected["quality"], cfg)
    if matched["student_indices"].numel() == 0:
        return zero, _positive_logs(zero)

    si, local_ti = matched["student_indices"], matched["teacher_indices"]
    source_ti = ti[local_ti]
    quality = selected["quality"][local_ti].detach()
    temp = max(cfg.temperature, cfg.eps)

    teacher_prob = torch.sigmoid(teacher_logits[source_ti] / temp).detach()
    cls_each = F.binary_cross_entropy_with_logits(
        student_logits[si] / temp, teacher_prob, reduction="none"
    ).mean(1) * temp * temp
    cls_loss = (cls_each * quality).sum() / quality.sum().clamp(min=cfg.eps)

    target = teacher_full[local_ti].detach()
    if cfg.box_loss_type == "smooth_l1":
        full_h, full_w = meta.full_hw
        norm = student_boxes_full.new_tensor([full_w, full_h, full_w, full_h]).clamp(min=1)
        box_each = F.smooth_l1_loss(
            student_boxes_full[si] / norm, target / norm, reduction="none"
        ).mean(1)
    elif cfg.box_loss_type == "giou":
        box_each = aligned_giou_loss_xyxy(student_boxes_full[si], target, cfg.eps)
    else:
        raise ValueError(f"unsupported box_loss_type: {cfg.box_loss_type}")
    box_loss = (box_each * quality).sum() / quality.sum().clamp(min=cfg.eps)

    ramp = student_logits.new_tensor(ramp_weight(epoch, cfg))
    total = ramp * (cfg.cls_weight * cls_loss + cfg.box_weight * box_loss)
    return total, {
        "loss_zvc_pos": total.detach(),
        "loss_zvc_cls": cls_loss.detach(),
        "loss_zvc_box": box_loss.detach(),
        "zvc_pos_pairs": student_logits.new_tensor(float(si.numel())),
        "zvc_quality_mean": quality.mean().detach(),
        "zvc_ramp": ramp.detach(),
    }


def _positive_logs(zero: Tensor) -> Dict[str, Tensor]:
    return {
        "loss_zvc_pos": zero.detach(),
        "loss_zvc_cls": zero.detach(),
        "loss_zvc_box": zero.detach(),
        "zvc_pos_pairs": zero.detach(),
        "zvc_quality_mean": zero.detach(),
        "zvc_ramp": zero.detach(),
    }


def reflective_hard_negative_loss(
    student_boxes: Tensor,
    student_logits: Tensor,
    meta: CropMeta,
    *,
    crop_has_gt: bool,
    teacher_max_conf: float,
    reflection_reliability: float,
    cfg: ZVCConfig,
    epoch: int,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    _check_boxes(student_boxes, "student_boxes")
    zero = student_logits.sum() * 0.0
    eligible = (
        not crop_has_gt
        and teacher_max_conf <= cfg.teacher_neg_conf_max
        and reflection_reliability > 0
    )
    if not eligible:
        return zero, _negative_logs(zero)

    inside = centers_inside_crop(student_boxes.detach(), meta)
    scores = student_logits.sigmoid().amax(1)
    candidates = torch.where(inside & (scores.detach() >= cfg.student_hardneg_conf))[0]
    if candidates.numel() == 0:
        return zero, _negative_logs(zero)

    k = min(cfg.max_hardneg_candidates_per_crop, int(candidates.numel()))
    chosen = candidates[torch.topk(scores[candidates].detach(), k=k).indices]
    each = F.binary_cross_entropy_with_logits(
        student_logits[chosen], torch.zeros_like(student_logits[chosen]), reduction="none"
    ).mean(1)
    reliability = student_logits.new_tensor(reflection_reliability).clamp(0, 1)
    loss = (
        student_logits.new_tensor(ramp_weight(epoch, cfg))
        * cfg.hardneg_weight * reliability * each.mean()
    )
    return loss, {
        "loss_zvc_hardneg": loss.detach(),
        "zvc_hardneg_count": student_logits.new_tensor(float(chosen.numel())),
        "zvc_hardneg_reliability": reliability.detach(),
    }


def _negative_logs(zero: Tensor) -> Dict[str, Tensor]:
    return {
        "loss_zvc_hardneg": zero.detach(),
        "zvc_hardneg_count": zero.detach(),
        "zvc_hardneg_reliability": zero.detach(),
    }


def combine_zvc_losses(positive_loss: Tensor, hard_negative_loss: Tensor) -> Tensor:
    return positive_loss + hard_negative_loss
