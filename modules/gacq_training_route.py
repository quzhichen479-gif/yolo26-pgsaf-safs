"""C1-GACQ: GT-Anchored Cross-View Quality Learning.

Reference implementation for a training-only auxiliary route.

Design constraints:
- shared C1 P3/P4/P5 features must be passed without detach;
- teacher predictions are used only to estimate reliability, never as box/class targets;
- localization targets remain ground-truth boxes;
- correct-class scores are trained to reflect localization quality;
- the auxiliary head and crop teacher must be removed for validation/export/deployment.

This module is intentionally detector-agnostic. Codex must adapt feature capture,
candidate assignment, and loss integration to the actual YOLO26-C1 repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class GACQConfig:
    """Conservative defaults for the first controlled experiments."""

    strides: Tuple[int, ...] = (8, 16, 32)
    level_size_thresholds: Tuple[float, ...] = (32.0, 96.0)
    candidate_radius: int = 1
    max_candidates_per_gt: int = 5

    teacher_conf_min: float = 0.35
    teacher_iou_min: float = 0.50
    visible_fraction_min: float = 0.85
    stability_min: float = 0.50
    student_evidence_min: float = 0.05

    lambda_loc: float = 1.0
    lambda_nwd: float = 0.5
    lambda_quality: float = 0.5
    lambda_rank: float = 0.10
    rank_margin: float = 0.05
    quality_gap_min: float = 0.05
    nwd_c: float = 16.0

    warmup_epochs: int = 5
    ramp_epochs: int = 15
    eps: float = 1e-7


@dataclass(frozen=True)
class CandidatePoint:
    """A fixed GT-anchored distillation point on one FPN level."""

    gt_index: int
    level_index: int
    y: int
    x: int
    stride: int


class ConvTower(nn.Module):
    """Small training-only tower."""

    def __init__(self, c1: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c1, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GACQAuxiliaryHead(nn.Module):
    """Training-only head operating on non-detached shared C1 features.

    Outputs per level:
      box_raw: [B, 4, H, W], decoded as positive l/t/r/b distances;
      cls_logits: [B, C, H, W];
      quality_logit: [B, 1, H, W].
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        num_classes: int,
        hidden_channels: int = 96,
    ) -> None:
        super().__init__()
        if len(in_channels) == 0:
            raise ValueError("in_channels must contain at least one feature level")
        self.num_classes = int(num_classes)
        self.towers = nn.ModuleList(ConvTower(int(c), hidden_channels) for c in in_channels)
        self.box_heads = nn.ModuleList(nn.Conv2d(hidden_channels, 4, 1) for _ in in_channels)
        self.cls_heads = nn.ModuleList(
            nn.Conv2d(hidden_channels, self.num_classes, 1) for _ in in_channels
        )
        self.quality_heads = nn.ModuleList(nn.Conv2d(hidden_channels, 1, 1) for _ in in_channels)

    def forward(self, features: Sequence[Tensor]) -> List[Dict[str, Tensor]]:
        if len(features) != len(self.towers):
            raise ValueError(
                f"expected {len(self.towers)} feature levels, got {len(features)}"
            )
        outputs: List[Dict[str, Tensor]] = []
        for feat, tower, box_head, cls_head, quality_head in zip(
            features, self.towers, self.box_heads, self.cls_heads, self.quality_heads
        ):
            h = tower(feat)
            outputs.append(
                {
                    "box_raw": box_head(h),
                    "cls_logits": cls_head(h),
                    "quality_logit": quality_head(h),
                }
            )
        return outputs


def box_area_xyxy(boxes: Tensor) -> Tensor:
    wh = (boxes[..., 2:4] - boxes[..., 0:2]).clamp(min=0.0)
    return wh[..., 0] * wh[..., 1]


def aligned_iou_xyxy(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    if pred.shape != target.shape or pred.shape[-1] != 4:
        raise ValueError("pred and target must have identical [...,4] shapes")
    lt = torch.maximum(pred[..., 0:2], target[..., 0:2])
    rb = torch.minimum(pred[..., 2:4], target[..., 2:4])
    inter_wh = (rb - lt).clamp(min=0.0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    union = box_area_xyxy(pred) + box_area_xyxy(target) - inter
    return inter / (union + eps)


def aligned_giou_loss_xyxy(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    iou = aligned_iou_xyxy(pred, target, eps)
    enc_lt = torch.minimum(pred[..., 0:2], target[..., 0:2])
    enc_rb = torch.maximum(pred[..., 2:4], target[..., 2:4])
    enc_area = box_area_xyxy(torch.cat([enc_lt, enc_rb], dim=-1))
    lt = torch.maximum(pred[..., 0:2], target[..., 0:2])
    rb = torch.minimum(pred[..., 2:4], target[..., 2:4])
    inter_wh = (rb - lt).clamp(min=0.0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    union = box_area_xyxy(pred) + box_area_xyxy(target) - inter
    giou = iou - (enc_area - union) / (enc_area + eps)
    return 1.0 - giou


def nwd_loss_xyxy(pred: Tensor, target: Tensor, c: float = 16.0, eps: float = 1e-7) -> Tensor:
    def to_cxcywh(box: Tensor) -> Tensor:
        center = (box[..., 0:2] + box[..., 2:4]) * 0.5
        wh = (box[..., 2:4] - box[..., 0:2]).clamp(min=eps)
        return torch.cat([center, wh], dim=-1)

    p = to_cxcywh(pred)
    t = to_cxcywh(target)
    dc = (p[..., 0:2] - t[..., 0:2]).pow(2).sum(dim=-1)
    ds = (p[..., 2:4] - t[..., 2:4]).pow(2).sum(dim=-1) / 4.0
    d = torch.sqrt((dc + ds).clamp(min=eps))
    return 1.0 - torch.exp(-d / max(float(c), eps))


def select_level_for_box(box_xyxy: Tensor, cfg: GACQConfig) -> int:
    """Select one FPN level using sqrt(box area) in input-image pixels."""
    wh = (box_xyxy[2:4] - box_xyxy[0:2]).clamp(min=cfg.eps)
    size = float(torch.sqrt(wh[0] * wh[1]).detach().cpu())
    for level, threshold in enumerate(cfg.level_size_thresholds):
        if size <= threshold:
            return level
    return min(len(cfg.strides) - 1, len(cfg.level_size_thresholds))


def build_gt_candidate_points(
    gt_boxes_xyxy: Tensor,
    feature_shapes: Sequence[Tuple[int, int]],
    cfg: GACQConfig,
) -> List[CandidatePoint]:
    """Build center-first fixed points around each GT."""
    if gt_boxes_xyxy.ndim != 2 or gt_boxes_xyxy.shape[-1] != 4:
        raise ValueError("gt_boxes_xyxy must be [N,4]")
    if len(feature_shapes) != len(cfg.strides):
        raise ValueError("feature_shapes and strides must have equal lengths")

    offsets = [(0, 0)]
    for d in range(1, cfg.candidate_radius + 1):
        offsets.extend([(0, -d), (0, d), (-d, 0), (d, 0)])
    for dy in range(-cfg.candidate_radius, cfg.candidate_radius + 1):
        for dx in range(-cfg.candidate_radius, cfg.candidate_radius + 1):
            if (dy, dx) not in offsets:
                offsets.append((dy, dx))

    points: List[CandidatePoint] = []
    for gt_idx, box in enumerate(gt_boxes_xyxy):
        level = select_level_for_box(box, cfg)
        stride = int(cfg.strides[level])
        h, w = feature_shapes[level]
        center = (box[0:2] + box[2:4]) * 0.5
        cx = int(torch.floor(center[0] / stride).item())
        cy = int(torch.floor(center[1] / stride).item())
        used = set()
        count = 0
        for dy, dx in offsets:
            y = max(0, min(h - 1, cy + dy))
            x = max(0, min(w - 1, cx + dx))
            if (y, x) in used:
                continue
            used.add((y, x))
            points.append(CandidatePoint(gt_idx, level, y, x, stride))
            count += 1
            if count >= cfg.max_candidates_per_gt:
                break
    return points


def gather_auxiliary_candidates(
    outputs: Sequence[Dict[str, Tensor]],
    points: Sequence[CandidatePoint],
    image_index: int,
) -> Dict[str, Tensor]:
    """Gather auxiliary predictions at fixed points for one image."""
    if not points:
        device = outputs[0]["box_raw"].device
        return {
            "box_raw": torch.zeros((0, 4), device=device),
            "cls_logits": torch.zeros((0, outputs[0]["cls_logits"].shape[1]), device=device),
            "quality_logit": torch.zeros((0,), device=device),
            "point_xy": torch.zeros((0, 2), device=device),
            "gt_indices": torch.zeros((0,), dtype=torch.long, device=device),
            "strides": torch.zeros((0,), device=device),
        }

    box_rows, cls_rows, q_rows, point_xy, gt_indices, strides = [], [], [], [], [], []
    for point in points:
        out = outputs[point.level_index]
        box_rows.append(out["box_raw"][image_index, :, point.y, point.x])
        cls_rows.append(out["cls_logits"][image_index, :, point.y, point.x])
        q_rows.append(out["quality_logit"][image_index, 0, point.y, point.x])
        device = out["box_raw"].device
        dtype = out["box_raw"].dtype
        point_xy.append(
            torch.tensor(
                [(point.x + 0.5) * point.stride, (point.y + 0.5) * point.stride],
                device=device,
                dtype=dtype,
            )
        )
        gt_indices.append(point.gt_index)
        strides.append(float(point.stride))

    ref = box_rows[0]
    return {
        "box_raw": torch.stack(box_rows),
        "cls_logits": torch.stack(cls_rows),
        "quality_logit": torch.stack(q_rows),
        "point_xy": torch.stack(point_xy),
        "gt_indices": torch.tensor(gt_indices, dtype=torch.long, device=ref.device),
        "strides": torch.tensor(strides, dtype=ref.dtype, device=ref.device),
    }


def decode_ltrb_boxes(box_raw: Tensor, point_xy: Tensor, strides: Tensor) -> Tensor:
    """Decode positive l/t/r/b distances into pixel xyxy boxes."""
    if box_raw.ndim != 2 or box_raw.shape[-1] != 4:
        raise ValueError("box_raw must be [N,4]")
    distances = F.softplus(box_raw) * strides[:, None]
    x, y = point_xy[:, 0], point_xy[:, 1]
    return torch.stack(
        [
            x - distances[:, 0],
            y - distances[:, 1],
            x + distances[:, 2],
            y + distances[:, 3],
        ],
        dim=-1,
    )


def teacher_reliability_weight(
    teacher_conf: Tensor,
    teacher_iou_to_gt: Tensor,
    visible_fraction: Tensor,
    stability: Optional[Tensor],
    student_evidence: Optional[Tensor],
    cfg: GACQConfig,
) -> Tuple[Tensor, Tensor]:
    """Build a detached geometric-mean teacher reliability."""
    values = [teacher_conf, teacher_iou_to_gt, visible_fraction]
    thresholds = [cfg.teacher_conf_min, cfg.teacher_iou_min, cfg.visible_fraction_min]

    if stability is None:
        stability = torch.ones_like(teacher_conf)
    if student_evidence is None:
        student_evidence = torch.ones_like(teacher_conf)
    values.extend([stability, student_evidence])
    thresholds.extend([cfg.stability_min, cfg.student_evidence_min])

    valid = torch.ones_like(teacher_conf, dtype=torch.bool)
    clipped = []
    for value, threshold in zip(values, thresholds):
        if value.shape != teacher_conf.shape:
            raise ValueError("all reliability tensors must share the same shape")
        valid &= value >= threshold
        clipped.append(value.detach().clamp(min=cfg.eps, max=1.0))

    stacked = torch.stack(clipped, dim=0)
    geometric_mean = torch.exp(torch.log(stacked).mean(dim=0))
    weight = torch.where(valid, geometric_mean, torch.zeros_like(geometric_mean))
    return weight.detach(), valid.detach()


def auxiliary_ramp(epoch: int, cfg: GACQConfig) -> float:
    if epoch < cfg.warmup_epochs:
        return 0.0
    if cfg.ramp_epochs <= 0:
        return 1.0
    return min(1.0, (epoch - cfg.warmup_epochs + 1) / cfg.ramp_epochs)


def pairwise_quality_ranking_loss(
    scores: Tensor,
    target_quality: Tensor,
    gt_indices: Tensor,
    cfg: GACQConfig,
) -> Tuple[Tensor, int]:
    """Rank candidates of the same GT according to localization quality."""
    losses: List[Tensor] = []
    pair_count = 0
    for gt_idx in torch.unique(gt_indices):
        mask = gt_indices == gt_idx
        s = scores[mask]
        q = target_quality[mask]
        if s.numel() < 2:
            continue
        qdiff = q[:, None] - q[None, :]
        valid = qdiff > cfg.quality_gap_min
        if not valid.any():
            continue
        ranking = F.relu(cfg.rank_margin - s[:, None] + s[None, :])
        weights = qdiff.clamp(min=0.0)
        losses.append((ranking[valid] * weights[valid]).mean())
        pair_count += int(valid.sum().item())
    if not losses:
        return scores.sum() * 0.0, 0
    return torch.stack(losses).mean(), pair_count


def gacq_auxiliary_loss(
    gathered: Dict[str, Tensor],
    gt_boxes_xyxy: Tensor,
    gt_labels: Tensor,
    gt_reliability: Tensor,
    cfg: GACQConfig,
    epoch: int,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Compute GT-anchored localization and quality-consistent ranking losses."""
    box_raw = gathered["box_raw"]
    if box_raw.numel() == 0:
        zero = box_raw.sum() * 0.0
        return zero, {
            "gacq_loss": zero.detach(),
            "gacq_loc": zero.detach(),
            "gacq_nwd": zero.detach(),
            "gacq_quality": zero.detach(),
            "gacq_rank": zero.detach(),
            "gacq_candidates": zero.detach(),
            "gacq_rank_pairs": zero.detach(),
            "gacq_ramp": zero.detach(),
        }

    gt_indices = gathered["gt_indices"]
    target_boxes = gt_boxes_xyxy[gt_indices]
    target_labels = gt_labels[gt_indices]
    reliability = gt_reliability[gt_indices].detach()

    pred_boxes = decode_ltrb_boxes(
        gathered["box_raw"], gathered["point_xy"], gathered["strides"]
    )
    iou = aligned_iou_xyxy(pred_boxes, target_boxes, cfg.eps)
    loc = aligned_giou_loss_xyxy(pred_boxes, target_boxes, cfg.eps)
    nwd = nwd_loss_xyxy(pred_boxes, target_boxes, cfg.nwd_c, cfg.eps)

    candidate_weight = reliability
    denom = candidate_weight.sum().clamp(min=cfg.eps)
    loss_loc = (loc * candidate_weight).sum() / denom
    loss_nwd = (nwd * candidate_weight).sum() / denom

    row = torch.arange(target_labels.numel(), device=target_labels.device)
    correct_class_logits = gathered["cls_logits"][row, target_labels]
    target_quality = iou.detach().clamp(0.0, 1.0)

    cls_quality = F.binary_cross_entropy_with_logits(
        correct_class_logits, target_quality, reduction="none"
    )
    quality_branch = F.binary_cross_entropy_with_logits(
        gathered["quality_logit"], target_quality, reduction="none"
    )
    loss_quality = ((cls_quality + quality_branch) * candidate_weight).sum() / denom

    scores = torch.sqrt(
        torch.sigmoid(correct_class_logits).clamp(min=cfg.eps)
        * torch.sigmoid(gathered["quality_logit"]).clamp(min=cfg.eps)
    )
    active_rank = reliability > 0
    loss_rank, rank_pairs = pairwise_quality_ranking_loss(
        scores=scores[active_rank],
        target_quality=target_quality[active_rank],
        gt_indices=gt_indices[active_rank],
        cfg=cfg,
    )

    ramp = auxiliary_ramp(epoch, cfg)
    total = ramp * (
        cfg.lambda_loc * loss_loc
        + cfg.lambda_nwd * loss_nwd
        + cfg.lambda_quality * loss_quality
        + cfg.lambda_rank * loss_rank
    )

    finite = torch.isfinite(total)
    if not bool(finite):
        total = box_raw.sum() * 0.0

    logs = {
        "gacq_loss": total.detach(),
        "gacq_loc": loss_loc.detach(),
        "gacq_nwd": loss_nwd.detach(),
        "gacq_quality": loss_quality.detach(),
        "gacq_rank": loss_rank.detach(),
        "gacq_iou_mean": iou.mean().detach(),
        "gacq_target_quality_mean": target_quality.mean().detach(),
        "gacq_reliability_mean": reliability.mean().detach(),
        "gacq_candidates": box_raw.new_tensor(float(box_raw.shape[0])),
        "gacq_rank_pairs": box_raw.new_tensor(float(rank_pairs)),
        "gacq_ramp": box_raw.new_tensor(float(ramp)),
        "gacq_finite": box_raw.new_tensor(float(bool(finite))),
    }
    return total, logs


def strip_auxiliary_state_dict(
    state_dict: Dict[str, Tensor],
    prefixes: Iterable[str] = ("gacq_aux", "crop_teacher", "gacq_teacher"),
) -> Dict[str, Tensor]:
    """Return a deployment state dict without training-only components."""
    prefix_tuple = tuple(prefixes)
    return {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(prefix_tuple)
    }
