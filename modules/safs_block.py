"""
SAFS-Block: Specular-Aware Spatial-Frequency Selection Block.

Intended usage:
    P3_refined = SAFSBlock(c)([P3_C1, spec])
where spec is optional [B,2,H0,W0], channel0=Rcore, channel1=Eedge.

This is a residual-safe prototype inspired by spatial-frequency selection ideas.
It is not a direct SFS-DETR copy. It is adapted to water-surface detection:
    - enhance useful local edge structures;
    - suppress specular-core and periodic-water interference;
    - keep Detect/loss/assignment unchanged.
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
    if spec is None:
        z = ref.new_zeros((ref.shape[0], 1, ref.shape[-2], ref.shape[-1]))
        return z, z
    if spec.dim() != 4 or spec.shape[1] < 2:
        raise ValueError("spec must be [B, >=2, H, W], channel0=Rcore, channel1=Eedge")
    spec = _resize_like(spec[:, :2], ref)
    rcore = spec[:, 0:1].clamp(0.0, 1.0)
    eedge = spec[:, 1:2].clamp(0.0, 1.0)
    return rcore, eedge


class SAFSBlock(nn.Module):
    """Specular-aware spatial-frequency selection block for P3 features."""
    def __init__(self, c: int, hidden_ratio: float = 0.5, freq_groups: int = 4):
        super().__init__()
        c_hidden = max(16, int(c * hidden_ratio))
        self.pre = ConvBNAct(c, c_hidden, k=1)

        # Spatial selection: local, horizontal, and vertical structures.
        self.local = ConvBNAct(c_hidden, c_hidden, k=3, g=c_hidden)
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
        self.spatial_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_hidden, 3, 1),
            nn.Softmax(dim=1),
        )

        # Frequency selection: real/imag 1x1 mixing in rFFT domain.
        # Keep groups conservative to avoid instability when c_hidden is small.
        self.freq_mix = nn.Conv2d(2 * c_hidden, 2 * c_hidden, kernel_size=1, groups=1, bias=True)
        self.freq_norm = nn.BatchNorm2d(c_hidden)

        self.post = ConvBNAct(c_hidden * 2, c, k=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.spec_ab = nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float32))

    def _frequency_branch(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        h, w = x.shape[-2:]
        xf = torch.fft.rfft2(x.float(), norm="ortho")
        ri = torch.cat([xf.real, xf.imag], dim=1)
        ri = self.freq_mix(ri)
        real, imag = ri.chunk(2, dim=1)
        yf = torch.complex(real, imag)
        y = torch.fft.irfft2(yf, s=(h, w), norm="ortho")
        y = self.freq_norm(y.to(orig_dtype))
        return F.silu(y, inplace=True)

    def forward(self, inputs: Union[Sequence[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            x, spec = inputs, None
        else:
            if len(inputs) < 1:
                raise ValueError("SAFSBlock expects P3 or [P3, optional_spec]")
            x = inputs[0]
            spec = inputs[1] if len(inputs) > 1 else None

        z = self.pre(x)
        branches = torch.stack([self.local(z), self.hstrip(z), self.vstrip(z)], dim=1)  # [B,3,C,H,W]
        sw = self.spatial_weight(z).unsqueeze(2)  # [B,3,1,1,1]
        z_spatial = (branches * sw).sum(dim=1)
        z_freq = self._frequency_branch(z)

        rcore, eedge = _split_spec(spec, x)
        edge_gain = torch.relu(self.spec_ab[0])
        core_penalty = torch.relu(self.spec_ab[1])
        spec_gate = torch.sigmoid(edge_gain * eedge - core_penalty * rcore)

        # Gate both spatial and frequency enhancement. Residual preserves C1 behavior at init.
        enhanced = self.post(torch.cat([z_spatial * spec_gate, z_freq * spec_gate], dim=1))
        return x + self.gamma * enhanced
