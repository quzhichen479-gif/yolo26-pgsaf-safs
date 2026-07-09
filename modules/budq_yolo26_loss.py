"""BUDQ-YOLO26 loss prototype.

This file is a Codex-ready reference implementation for adapting BUDQ-YOLO26
into the actual YOLO26-C1 repository.

Important:
- This prototype does NOT add DFL.
- This prototype does NOT add reg_max.
- This prototype does NOT modify Detect.
- It works on continuous boxes in xyxy format.
- Ranking helpers are optional and should be integrated only after UBR is stable.

Expected integration point in the actual YOLO26 repo:
- adapt these helpers into the existing detection loss file;
- use the repository's existing matched positive predictions and targets;
- keep assignment unchanged;
- keep the original classification loss unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class BUDQConfig:
    """Hyperparameters for BUDQ-YOLO26.

    Values are intentionally conservative. Use the Y-series matrix before tuning.
    """

    tau: float = 32.0
    nwd_c: float = 16.0
    rho: float = 0.10
    u_min: float = 1.0
    u_max: float = 4.0
    tight_warmup_epochs: int = 30
    lambda_cover: float = 1.0
    lambda_spill: float = 0.5
    lambda_nwd: float = 1.0
    lambda_mpd: float = 0.5
    rank_margin: float = 0.10
    dup_margin: float = 0.05
    pos_iou_thr: float = 0.50
    neg_iou_thr: float = 0.25
    rank_topk_neg: int = 32
    lambda_rank_posneg: float = 0.2
    lambda_rank_dup: float = 0.1
    eps: float = 1e-7


def _box_area_xyxy(boxes: Tensor, eps: float = 1e-7) -> Tensor:
    """Area of xyxy boxes."""
    wh = (boxes[..., 2:4] - boxes[..., 0:2]).clamp(min=0.0)
    return wh[..., 0] * wh[..., 1] + eps


def _intersection_area_xyxy(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """Elementwise intersection area for aligned xyxy boxes."""
    lt = torch.maximum(boxes1[..., 0:2], boxes2[..., 0:2])
    rb = torch.minimum(boxes1[..., 2:4], boxes2[..., 2:4])
    wh = (rb - lt).clamp(min=0.0)
    return wh[..., 0] * wh[..., 1] + eps


def box_iou_aligned_xyxy(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """Elementwise IoU for aligned xyxy boxes with the same shape [N, 4]."""
    inter = _intersection_area_xyxy(pred, target, eps=0.0)
    area_p = _box_area_xyxy(pred, eps=0.0)
    area_t = _box_area_xyxy(target, eps=0.0)
    return inter / (area_p + area_t - inter + eps)


def xyxy_to_cxcywh(boxes: Tensor, eps: float = 1e-7) -> Tensor:
    """Convert xyxy boxes to cxcywh."""
    cxcy = (boxes[..., 0:2] + boxes[..., 2:4]) * 0.5
    wh = (boxes[..., 2:4] - boxes[..., 0:2]).clamp(min=eps)
    return torch.cat([cxcy, wh], dim=-1)


def boundary_uncertainty_boxes(
    target_xyxy: Tensor,
    image_size: Optional[Tuple[float, float]] = None,
    rho: float = 0.10,
    u_min: float = 1.0,
    u_max: float = 4.0,
    eps: float = 1e-7,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Build inner/outer boxes for boundary uncertainty.

    Args:
        target_xyxy: target boxes, shape [N, 4]. Coordinates may be pixels or normalized.
        image_size: optional (height, width). If None, u_min/u_max are assumed to be in the
            same coordinate units as target_xyxy. If target boxes are normalized, pass image_size
            and convert u manually before calling, or set u_min/u_max in normalized units.
        rho: uncertainty radius ratio.
        u_min: minimum uncertainty radius.
        u_max: maximum uncertainty radius.

    Returns:
        inner_box, outer_box, u.
    """
    del image_size  # Kept for integration clarity; actual repo may need coordinate conversion.
    cxcywh = xyxy_to_cxcywh(target_xyxy, eps=eps)
    wh = cxcywh[..., 2:4]
    scale = torch.sqrt((wh[..., 0] * wh[..., 1]).clamp(min=eps))
    u = torch.clamp(rho * scale, min=u_min, max=u_max)

    inner = target_xyxy.clone()
    inner[..., 0] = target_xyxy[..., 0] + u
    inner[..., 1] = target_xyxy[..., 1] + u
    inner[..., 2] = target_xyxy[..., 2] - u
    inner[..., 3] = target_xyxy[..., 3] - u

    # If a tiny box collapses after shrink, fall back to a one-point core around center.
    center = cxcywh[..., 0:2]
    inner_w = (inner[..., 2] - inner[..., 0]).clamp(min=0.0)
    inner_h = (inner[..., 3] - inner[..., 1]).clamp(min=0.0)
    collapsed = (inner_w <= eps) | (inner_h <= eps)
    if collapsed.any():
        half = torch.full_like(u, fill_value=0.5)
        inner[..., 0] = torch.where(collapsed, center[..., 0] - half, inner[..., 0])
        inner[..., 1] = torch.where(collapsed, center[..., 1] - half, inner[..., 1])
        inner[..., 2] = torch.where(collapsed, center[..., 0] + half, inner[..., 2])
        inner[..., 3] = torch.where(collapsed, center[..., 1] + half, inner[..., 3])

    outer = target_xyxy.clone()
    outer[..., 0] = target_xyxy[..., 0] - u
    outer[..., 1] = target_xyxy[..., 1] - u
    outer[..., 2] = target_xyxy[..., 2] + u
    outer[..., 3] = target_xyxy[..., 3] + u

    return inner, outer, u


def core_coverage_loss(pred_xyxy: Tensor, inner_xyxy: Tensor, eps: float = 1e-7) -> Tensor:
    """1 - fraction of inner core covered by the predicted box."""
    inter = _intersection_area_xyxy(pred_xyxy, inner_xyxy, eps=0.0)
    inner_area = _box_area_xyxy(inner_xyxy, eps=0.0)
    return 1.0 - inter / (inner_area + eps)


def spill_loss(pred_xyxy: Tensor, outer_xyxy: Tensor, eps: float = 1e-7) -> Tensor:
    """Fraction of predicted area outside the tolerated outer box."""
    pred_area = _box_area_xyxy(pred_xyxy, eps=0.0)
    inter_outer = _intersection_area_xyxy(pred_xyxy, outer_xyxy, eps=0.0)
    spill = (pred_area - inter_outer).clamp(min=0.0)
    return spill / (pred_area + eps)


def nwd_loss_xyxy(pred_xyxy: Tensor, target_xyxy: Tensor, c: float = 16.0, eps: float = 1e-7) -> Tensor:
    """NWD-style continuous box distance loss.

    The coordinate scale must match c. If boxes are normalized, adapt c accordingly.
    """
    p = xyxy_to_cxcywh(pred_xyxy, eps=eps)
    t = xyxy_to_cxcywh(target_xyxy, eps=eps)
    dc = (p[..., 0:2] - t[..., 0:2]).pow(2).sum(dim=-1)
    ds = (p[..., 2:4] - t[..., 2:4]).pow(2).sum(dim=-1) / 4.0
    d = torch.sqrt((dc + ds).clamp(min=eps))
    return 1.0 - torch.exp(-d / max(c, eps))


def small_object_weight(target_xyxy: Tensor, tau: float = 32.0, eps: float = 1e-7) -> Tensor:
    """alpha_s = exp(-sqrt(w*h)/tau)."""
    wh = (target_xyxy[..., 2:4] - target_xyxy[..., 0:2]).clamp(min=eps)
    scale = torch.sqrt((wh[..., 0] * wh[..., 1]).clamp(min=eps))
    return torch.exp(-scale / max(tau, eps)).clamp(0.0, 1.0)


def mpdiou_loss_xyxy(pred_xyxy: Tensor, target_xyxy: Tensor, eps: float = 1e-7) -> Tensor:
    """MPDIoU-style corner-distance IoU loss for aligned xyxy boxes.

    This is a conservative helper. If the actual repo already has MPDIoU, prefer the repo's
    official implementation to avoid duplicated behavior.
    """
    iou = box_iou_aligned_xyxy(pred_xyxy, target_xyxy, eps=eps)

    d_tl = (pred_xyxy[..., 0:2] - target_xyxy[..., 0:2]).pow(2).sum(dim=-1)
    d_br = (pred_xyxy[..., 2:4] - target_xyxy[..., 2:4]).pow(2).sum(dim=-1)

    enc_lt = torch.minimum(pred_xyxy[..., 0:2], target_xyxy[..., 0:2])
    enc_rb = torch.maximum(pred_xyxy[..., 2:4], target_xyxy[..., 2:4])
    enc_wh = (enc_rb - enc_lt).clamp(min=eps)
    norm = enc_wh[..., 0].pow(2) + enc_wh[..., 1].pow(2) + eps

    return 1.0 - iou + (d_tl + d_br) / norm


def ubr_box_loss(
    pred_xyxy: Tensor,
    target_xyxy: Tensor,
    cfg: BUDQConfig,
    epoch: int = 0,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Compute BUDQ UBR box loss for matched positive boxes.

    Args:
        pred_xyxy: matched predicted boxes, shape [N, 4].
        target_xyxy: matched target boxes, shape [N, 4].
        cfg: BUDQConfig.
        epoch: current epoch for delayed MPDIoU tightening.

    Returns:
        loss: scalar mean loss.
        components: detached-ish component dictionary for logging.
    """
    if pred_xyxy.numel() == 0:
        zero = pred_xyxy.sum() * 0.0
        return zero, {
            "loss_ubr": zero,
            "loss_cover": zero,
            "loss_spill": zero,
            "loss_nwd": zero,
            "loss_mpdiou": zero,
            "alpha_s_mean": zero,
            "beta_t": zero,
        }

    inner, outer, _ = boundary_uncertainty_boxes(
        target_xyxy,
        rho=cfg.rho,
        u_min=cfg.u_min,
        u_max=cfg.u_max,
        eps=cfg.eps,
    )
    alpha = small_object_weight(target_xyxy, tau=cfg.tau, eps=cfg.eps)
    beta_t_value = min(1.0, float(epoch) / max(1, cfg.tight_warmup_epochs))
    beta_t = pred_xyxy.new_tensor(beta_t_value)

    l_cover = core_coverage_loss(pred_xyxy, inner, eps=cfg.eps)
    l_spill = spill_loss(pred_xyxy, outer, eps=cfg.eps)
    l_nwd = nwd_loss_xyxy(pred_xyxy, target_xyxy, c=cfg.nwd_c, eps=cfg.eps)
    l_mpd = mpdiou_loss_xyxy(pred_xyxy, target_xyxy, eps=cfg.eps)

    ubr_each = alpha * (
        cfg.lambda_cover * l_cover
        + cfg.lambda_spill * l_spill
        + cfg.lambda_nwd * l_nwd
    ) + beta_t * cfg.lambda_mpd * l_mpd

    loss = ubr_each.mean()
    components = {
        "loss_ubr": loss.detach(),
        "loss_cover": l_cover.mean().detach(),
        "loss_spill": l_spill.mean().detach(),
        "loss_nwd": l_nwd.mean().detach(),
        "loss_mpdiou": l_mpd.mean().detach(),
        "alpha_s_mean": alpha.mean().detach(),
        "beta_t": beta_t.detach(),
    }
    return loss, components


def pairwise_posneg_ranking_loss(
    pos_scores: Tensor,
    neg_scores: Tensor,
    margin: float = 0.10,
    topk_neg: int = 32,
) -> Tensor:
    """Pairwise ranking: positive scores should outrank high-score negatives.

    Args:
        pos_scores: scores for reliable positives, shape [P].
        neg_scores: scores for reliable negatives/backgrounds, shape [N].
        margin: ranking margin.
        topk_neg: use only top-k negatives for stability.
    """
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return (pos_scores.sum() + neg_scores.sum()) * 0.0

    k = min(int(topk_neg), int(neg_scores.numel()))
    hard_neg = torch.topk(neg_scores, k=k, largest=True).values
    # [P, K]
    loss = F.relu(margin - pos_scores[:, None] + hard_neg[None, :])
    return loss.mean()


def duplicate_ranking_loss(
    scores: Tensor,
    quality: Tensor,
    gt_ids: Tensor,
    margin: float = 0.05,
) -> Tensor:
    """Rank the best candidate for each GT above its duplicate candidates.

    Args:
        scores: candidate confidence/class scores, shape [N].
        quality: localization quality for candidates, e.g. IoU or NWD similarity, shape [N].
        gt_ids: assigned GT id for each candidate, shape [N]. Candidates with gt_id < 0 are ignored.
    """
    if scores.numel() == 0:
        return scores.sum() * 0.0

    valid = gt_ids >= 0
    if valid.sum() == 0:
        return scores.sum() * 0.0

    losses = []
    unique_ids = torch.unique(gt_ids[valid])
    for gid in unique_ids:
        idx = torch.nonzero((gt_ids == gid), as_tuple=False).flatten()
        if idx.numel() <= 1:
            continue
        q = quality[idx]
        leader_local = torch.argmax(q)
        leader_idx = idx[leader_local]
        dup_idx = idx[idx != leader_idx]
        if dup_idx.numel() == 0:
            continue
        l = F.relu(margin - scores[leader_idx] + scores[dup_idx])
        losses.append(l.mean())

    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def dar_ranking_loss(
    pos_scores: Tensor,
    neg_scores: Tensor,
    dup_scores: Optional[Tensor] = None,
    dup_quality: Optional[Tensor] = None,
    dup_gt_ids: Optional[Tensor] = None,
    cfg: Optional[BUDQConfig] = None,
    enable_dup: bool = False,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Compute BUDQ duplicate-aware ranking loss.

    The actual repository should build pos_scores/neg_scores from its existing matched outputs.
    Do not modify assignment for Y0-Y5.
    """
    cfg = cfg or BUDQConfig()
    l_posneg = pairwise_posneg_ranking_loss(
        pos_scores,
        neg_scores,
        margin=cfg.rank_margin,
        topk_neg=cfg.rank_topk_neg,
    )

    if enable_dup and dup_scores is not None and dup_quality is not None and dup_gt_ids is not None:
        l_dup = duplicate_ranking_loss(
            dup_scores,
            dup_quality,
            dup_gt_ids,
            margin=cfg.dup_margin,
        )
    else:
        l_dup = l_posneg * 0.0

    total = cfg.lambda_rank_posneg * l_posneg + cfg.lambda_rank_dup * l_dup
    return total, {
        "loss_rank_total": total.detach(),
        "loss_rank_posneg": l_posneg.detach(),
        "loss_rank_dup": l_dup.detach(),
    }


__all__ = [
    "BUDQConfig",
    "box_iou_aligned_xyxy",
    "boundary_uncertainty_boxes",
    "core_coverage_loss",
    "spill_loss",
    "nwd_loss_xyxy",
    "small_object_weight",
    "mpdiou_loss_xyxy",
    "ubr_box_loss",
    "pairwise_posneg_ranking_loss",
    "duplicate_ranking_loss",
    "dar_ranking_loss",
]
