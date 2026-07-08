"""
PG-SAF: Pspec-Guided Semantic-Aligned Fusion for YOLO-style necks.

Intended usage:
    F3_refined = PGSAF(c3, c4, c_out)([P3, P4, spec])
where:
    P3:  high-resolution small-object feature, shape [B, C3, H, W]
    P4:  deeper semantic feature, shape [B, C4, H/2, W/2] or already upsampled
    spec: optional two-channel prior [B, 2, H0, W0], channel 0 = Rcore, channel 1 = Eedge.
          Rcore and Eedge should be normalized to [0, 1].

Design constraints:
    - Do not change Detect / loss / assignment.
    - Residual-safe: final residual scale gamma is initialized to 0.
    - If spec is unavailable, the module falls back to content-only alignment.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

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


class PGSAF(nn.Module):
    """Pspec-Guided Semantic-Aligned Fusion.

    This module fuses P4 semantics into P3 while reducing specular-core contamination.
    It is intentionally light and residual-safe for drop-in YOLO experiments.
    """
    def __init__(self, c3: int, c4: int, c_out: Optional[int] = None, reduction: float = 0.5):
        super().__init__()
        c_out = c_out or c3
        c_mid = max(16, int(c_out * reduction))

        self.p3_proj = ConvBNAct(c3, c_mid, k=1)
        self.p4_proj = ConvBNAct(c4, c_mid, k=1)

        # Spatial refined alignment: content gate decides where P4 semantics should enter P3.
        self.align_gate = nn.Sequential(
            ConvBNAct(c_mid * 2, c_mid, k=3),
            nn.Conv2d(c_mid, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Channel semantic alignment.
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_mid * 2, max(8, c_mid // 4), 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(8, c_mid // 4), c_mid * 2, 1),
            nn.Sigmoid(),
        )

        self.fuse = ConvBNAct(c_mid * 2, c_out, k=3)
        self.shortcut = ConvBNAct(c3, c_out, k=1) if c3 != c_out else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros(1))

        # Learnable but bounded weights for specular prior influence.
        self.spec_ab = nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float32))  # [edge_gain, core_penalty]

    def forward(self, inputs: Union[Sequence[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            raise ValueError("PGSAF expects [P3, P4, optional_spec]")
        if len(inputs) < 2:
            raise ValueError("PGSAF expects at least [P3, P4]")

        p3, p4 = inputs[0], inputs[1]
        spec = inputs[2] if len(inputs) > 2 else None
        p4 = _resize_like(p4, p3)

        f3 = self.p3_proj(p3)
        f4 = self.p4_proj(p4)

        concat = torch.cat([f3, f4], dim=1)
        ch = self.channel_gate(concat)
        ch3, ch4 = ch.chunk(2, dim=1)
        f3a = f3 * ch3
        f4a = f4 * ch4

        content_align = self.align_gate(torch.cat([f3a, f4a], dim=1))
        rcore, eedge = _split_spec(spec, p3)
        edge_gain = torch.relu(self.spec_ab[0])
        core_penalty = torch.relu(self.spec_ab[1])
        spec_gate = torch.sigmoid(edge_gain * eedge - core_penalty * rcore)

        # P4 semantic injection is allowed at useful edges and suppressed at specular cores.
        aligned_f4 = f4a * content_align * spec_gate
        fused = self.fuse(torch.cat([f3a, aligned_f4], dim=1))
        return self.shortcut(p3) + self.gamma * fused
