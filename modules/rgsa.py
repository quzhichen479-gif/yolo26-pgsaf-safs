"""
RGSA: Reflectance-Guided Selective Attention for YOLO26-C1.

This module is a conservative, residual-safe prototype that turns the
"select crop" idea into an in-network spatial selection attention.

Core idea:
    select crop = region selection + image-space zoom
    RGSA only distills the region-selection part into P3 attention.

Intended usage:
    P3_refined = RGSA(c)([P3_C1, spec])

where:
    P3_C1: C1-refined P3 feature, shape [B, C, H, W]
    spec: optional two-channel prior [B, 2, H0, W0]
          channel 0 = Rcore, specular-core risk, normalized [0, 1]
          channel 1 = Eedge, useful specular/target edge cue, normalized [0, 1]

Design constraints:
    - Do not change Detect / loss / assignment by default.
    - Residual-safe: gamma is initialized to 0, so the initial model behaves like C1.
    - Selection is sparse and specular-aware: enhance Eedge-high regions, suppress Rcore-high regions.
    - It does not claim to replace RAS-v2c image-space zoom; it tests whether crop-selection knowledge helps full-image inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: Optional[int] = None, g: int = 1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


def _split_spec(spec: Optional[torch.Tensor], ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return normalized Rcore and Eedge maps resized to ref."""
    if spec is None:
        z = ref.new_zeros((ref.shape[0], 1, ref.shape[-2], ref.shape[-1]))
        return z, z
    if spec.dim() != 4 or spec.shape[1] < 2:
        raise ValueError("spec must be [B, >=2, H, W], channel0=Rcore, channel1=Eedge")
    spec = _resize_like(spec[:, :2], ref)
    rcore = spec[:, 0:1].clamp(0.0, 1.0)
    eedge = spec[:, 1:2].clamp(0.0, 1.0)
    return rcore, eedge


class RGSA(nn.Module):
    """Reflectance-Guided Selective Attention.

    RGSA refines P3 only. It predicts a sparse selection map S that should mimic
    where a reflectance-aware crop selector would spend zoom budget, then applies
    a bounded local residual enhancement only at selected positions.
    """

    def __init__(
        self,
        c: int,
        hidden_ratio: float = 0.5,
        max_gain: float = 0.35,
        use_strip: bool = True,
        return_attention: bool = False,
    ):
        super().__init__()
        c_hidden = max(16, int(c * hidden_ratio))
        self.max_gain = float(max_gain)
        self.use_strip = bool(use_strip)
        self.return_attention = bool(return_attention)

        self.pre = ConvBNAct(c, c_hidden, k=1)

        # Lightweight local refinement. This is not feature super-resolution;
        # it only sharpens already-present local evidence.
        self.local = ConvBNAct(c_hidden, c_hidden, k=3, g=c_hidden)
        if self.use_strip:
            self.hstrip = nn.Sequential(
                nn.Conv2d(c_hidden, c_hidden, kernel_size=(1, 7), padding=(0, 3), groups=c_hidden, bias=False),
                nn.BatchNorm2d(c_hidden),
                nn.SiLU(inplace=True),
            )
            self.vstrip = nn.Sequential(
                nn.Conv2d(c_hidden, c_hidden, kernel_size=(7, 1), padding=(3, 0), groups=c_hidden, bias=False),
                nn.BatchNorm2d(c_hidden),
                nn.SiLU(inplace=True),
            )
            refine_in = c_hidden * 3
        else:
            refine_in = c_hidden

        self.refine = ConvBNAct(refine_in, c, k=1)

        # Selection map head. It sees compressed P3 evidence and spec cues.
        # Inputs: projected P3, channel mean, channel max, Rcore, Eedge, Eedge*(1-Rcore).
        select_in = c_hidden + 5
        self.select_head = nn.Sequential(
            ConvBNAct(select_in, c_hidden, k=3),
            ConvBNAct(c_hidden, max(8, c_hidden // 2), k=3),
            nn.Conv2d(max(8, c_hidden // 2), 1, kernel_size=1, bias=True),
        )

        # Bounded specular prior influence. Positive edge_gain allows useful edge regions;
        # positive core_penalty suppresses isolated specular cores.
        self.spec_ab = nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float32))

        # gamma=0 makes the initial network an identity wrapper around C1.
        self.gamma = nn.Parameter(torch.zeros(1))

        # Debug buffers for optional diagnostics. They are intentionally not persistent.
        self.last_select: Optional[torch.Tensor] = None
        self.last_spec_gate: Optional[torch.Tensor] = None

    def _build_selection_inputs(self, z: torch.Tensor, rcore: torch.Tensor, eedge: torch.Tensor) -> torch.Tensor:
        mean_map = z.mean(dim=1, keepdim=True)
        max_map = z.amax(dim=1, keepdim=True)
        edge_not_core = eedge * (1.0 - rcore)
        risk_balance = eedge - rcore
        return torch.cat([z, mean_map, max_map, rcore, eedge, edge_not_core, risk_balance], dim=1)

    def _local_refine(self, z: torch.Tensor) -> torch.Tensor:
        z_local = self.local(z)
        if not self.use_strip:
            return self.refine(z_local)
        z_h = self.hstrip(z)
        z_v = self.vstrip(z)
        return self.refine(torch.cat([z_local, z_h, z_v], dim=1))

    def forward(self, inputs: Union[Sequence[torch.Tensor], torch.Tensor]) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if isinstance(inputs, torch.Tensor):
            x, spec = inputs, None
        else:
            if len(inputs) < 1:
                raise ValueError("RGSA expects P3 or [P3, optional_spec]")
            x = inputs[0]
            spec = inputs[1] if len(inputs) > 1 else None

        z = self.pre(x)
        rcore, eedge = _split_spec(spec, x)
        rcore_z = _resize_like(rcore, z)
        eedge_z = _resize_like(eedge, z)

        select_logits = self.select_head(self._build_selection_inputs(z, rcore_z, eedge_z))
        raw_select = torch.sigmoid(select_logits)

        edge_gain = torch.relu(self.spec_ab[0])
        core_penalty = torch.relu(self.spec_ab[1])
        spec_gate = torch.sigmoid(edge_gain * eedge - core_penalty * rcore)

        select = raw_select * spec_gate
        enhanced = self._local_refine(z)

        # Bounded residual: max_gain prevents attention from dominating C1 early training.
        gain = torch.tanh(self.gamma) * self.max_gain
        out = x + gain * select * enhanced

        self.last_select = select.detach()
        self.last_spec_gate = spec_gate.detach()

        if self.return_attention:
            return out, {"select": select, "raw_select": raw_select, "spec_gate": spec_gate, "select_logits": select_logits}
        return out


@dataclass
class RGSALossWeights:
    select: float = 0.05
    sparse: float = 0.005
    core_suppress: float = 0.02


class RGSAuxLoss(nn.Module):
    """Optional auxiliary loss for RGSA selection supervision.

    Codex can attach this only if the training loop already supports custom auxiliary losses.
    If integration is risky, skip this loss and first run RGSA as a pure residual module.

    Expected tensors:
        select: [B,1,H,W], sigmoid selection map from RGSA
        target: [B,1,H,W], pseudo select target in [0,1]
        rcore:  [B,1,H,W], optional core-risk map in [0,1]
    """

    def __init__(self, weights: RGSALossWeights = RGSALossWeights()):
        super().__init__()
        self.weights = weights

    def forward(
        self,
        select: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        rcore: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = select.new_tensor(0.0)
        if target is not None and self.weights.select > 0:
            target = _resize_like(target.clamp(0.0, 1.0), select)
            pos = target.detach()
            # Slight positive weighting because selected regions are sparse.
            weight = 1.0 + 4.0 * pos
            loss = loss + self.weights.select * F.binary_cross_entropy(select.clamp(1e-4, 1 - 1e-4), target, weight=weight)
        if self.weights.sparse > 0:
            loss = loss + self.weights.sparse * select.mean()
        if rcore is not None and self.weights.core_suppress > 0:
            rcore = _resize_like(rcore.clamp(0.0, 1.0), select)
            loss = loss + self.weights.core_suppress * (select * rcore).mean()
        return loss


def build_select_target_from_boxes(
    boxes_xyxy: torch.Tensor,
    batch_indices: torch.Tensor,
    batch_size: int,
    out_hw: Tuple[int, int],
    image_hw: Tuple[int, int],
    device: Optional[torch.device] = None,
    small_area_ratio: float = 0.04,
    expand: float = 1.5,
) -> torch.Tensor:
    """Build a simple small-object select target from normalized or pixel xyxy boxes.

    This helper is intentionally generic. Codex should adapt it to the actual YOLO26
    label format. Only small boxes are marked, expanded around their centers.
    """
    h, w = out_hw
    ih, iw = image_hw
    device = device or boxes_xyxy.device
    target = torch.zeros((batch_size, 1, h, w), device=device)
    if boxes_xyxy.numel() == 0:
        return target

    boxes = boxes_xyxy.to(device).float().clone()
    if boxes.max() <= 1.5:
        boxes[:, [0, 2]] *= iw
        boxes[:, [1, 3]] *= ih

    for box, bi in zip(boxes, batch_indices.to(device).long()):
        x1, y1, x2, y2 = box.tolist()
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        area_ratio = (bw * bh) / float(max(1, iw * ih))
        if area_ratio > small_area_ratio:
            continue
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bw *= expand
        bh *= expand
        gx1 = int(max(0, min(w - 1, (cx - bw / 2) / iw * w)))
        gy1 = int(max(0, min(h - 1, (cy - bh / 2) / ih * h)))
        gx2 = int(max(0, min(w - 1, (cx + bw / 2) / iw * w)))
        gy2 = int(max(0, min(h - 1, (cy + bh / 2) / ih * h)))
        target[int(bi), :, gy1 : gy2 + 1, gx1 : gx2 + 1] = 1.0
    return target
