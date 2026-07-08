"""
PGW-SAFR: Pseudo-specular Guided Wavelet Specular-Aware Feature Recalibration.

This module is intended as a conservative C1-internal ablation:
    P3_wave = PGWSAFR(c)([P3_C1, spec_prior])
    Detect([P3_wave, P4_C1, P5_C1])

where spec_prior is optional [B, 2, H0, W0]:
    channel 0 = Rcore, specular-core risk, normalized [0, 1]
    channel 1 = Eedge, useful specular/target edge cue, normalized [0, 1]

Design constraints:
    - no Detect/loss/assignment changes;
    - no image-space crop or two-pass inference;
    - no external wavelet dependency;
    - residual-safe gamma initialized to 0, so the initial model behaves like C1;
    - HH/diagonal high-frequency is never positively boosted by default because it is
      most likely to contain water-ripple, foam, and specular noise.
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


def _pad_to_even(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad feature map on right/bottom so Haar DWT can safely downsample by 2."""
    h, w = x.shape[-2:]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, (h, w)


def _haar_dwt2(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """Dependency-free orthonormal 2D Haar DWT."""
    x, orig_hw = _pad_to_even(x)
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, lh, hl, hh, orig_hw


def _haar_idwt2(
    ll: torch.Tensor,
    lh: torch.Tensor,
    hl: torch.Tensor,
    hh: torch.Tensor,
    orig_hw: Tuple[int, int],
) -> torch.Tensor:
    """Inverse of _haar_dwt2."""
    b, c, h, w = ll.shape
    y = ll.new_empty((b, c, h * 2, w * 2))

    y[:, :, 0::2, 0::2] = (ll + lh + hl + hh) * 0.5
    y[:, :, 0::2, 1::2] = (ll - lh + hl - hh) * 0.5
    y[:, :, 1::2, 0::2] = (ll + lh - hl - hh) * 0.5
    y[:, :, 1::2, 1::2] = (ll - lh - hl + hh) * 0.5

    h0, w0 = orig_hw
    return y[:, :, :h0, :w0]


class PGWSAFR(nn.Module):
    """Pseudo-specular guided wavelet feature recalibration.

    Recommended first ablations:
        - mode="veto": only suppress high-frequency response in specular-core regions.
        - mode="safr": suppress specular-core high frequency and lightly preserve Eedge bands.

    Args:
        c: input/output channels.
        hidden_ratio: internal channel ratio.
        mode: "veto" or "safr".
        max_edge_gain: upper bound for LH/HL edge-band gain.
        max_core_penalty: upper bound for high-frequency penalty inside Rcore.
        boost_hh: if True, HH may receive Eedge gain. Default False for water scenes.
    """
    def __init__(
        self,
        c: int,
        hidden_ratio: float = 0.5,
        mode: str = "safr",
        max_edge_gain: float = 0.12,
        max_core_penalty: float = 0.25,
        boost_hh: bool = False,
    ):
        super().__init__()
        if mode not in {"veto", "safr"}:
            raise ValueError('PGWSAFR mode must be "veto" or "safr"')

        c_hidden = max(16, int(c * hidden_ratio))
        self.mode = mode
        self.max_edge_gain = float(max_edge_gain)
        self.max_core_penalty = float(max_core_penalty)
        self.boost_hh = bool(boost_hh)

        self.pre = ConvBNAct(c, c_hidden, k=1)

        # Low-frequency context is processed lightly; no specular amplification is applied here.
        self.low_mix = nn.Sequential(
            nn.Conv2d(c_hidden, c_hidden, kernel_size=3, padding=1, groups=c_hidden, bias=False),
            nn.BatchNorm2d(c_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_hidden, c_hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_hidden),
            nn.SiLU(inplace=True),
        )

        # Band-wise content calibration before applying specular gates.
        self.high_mix = nn.Sequential(
            nn.Conv2d(3 * c_hidden, 3 * c_hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(3 * c_hidden),
            nn.SiLU(inplace=True),
        )

        self.post = ConvBNAct(c_hidden, c, k=1)

        # gamma=0 makes this a strict no-op at initialization after the residual path.
        self.gamma = nn.Parameter(torch.zeros(1))

        # Bounded non-negative strengths. Sigmoid(params) * max_* keeps early training stable.
        self.edge_gain_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.core_penalty_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def _gates(self, spec: Optional[torch.Tensor], ref_band: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rcore, eedge = _split_spec(spec, ref_band)
        edge_gain = torch.sigmoid(self.edge_gain_logit) * self.max_edge_gain
        core_penalty = torch.sigmoid(self.core_penalty_logit) * self.max_core_penalty

        # Eedge helps only outside high-risk cores; Rcore suppresses all high bands.
        edge_term = eedge * (1.0 - rcore)
        core_term = rcore

        if self.mode == "veto":
            lh_hl_gate = 1.0 - core_penalty * core_term
            hh_gate = lh_hl_gate
        else:
            lh_hl_gate = 1.0 + edge_gain * edge_term - core_penalty * core_term
            if self.boost_hh:
                hh_gate = lh_hl_gate
            else:
                hh_gate = 1.0 - core_penalty * core_term

        return lh_hl_gate.clamp(0.0, 1.0 + self.max_edge_gain), hh_gate.clamp(0.0, 1.0 + self.max_edge_gain)

    def forward(self, inputs: Union[Sequence[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            x, spec = inputs, None
        else:
            if len(inputs) < 1:
                raise ValueError("PGWSAFR expects P3/P4 feature or [feature, optional_spec]")
            x = inputs[0]
            spec = inputs[1] if len(inputs) > 1 else None

        z = self.pre(x)
        ll, lh, hl, hh, orig_hw = _haar_dwt2(z)

        ll = self.low_mix(ll)
        lh, hl, hh = self.high_mix(torch.cat([lh, hl, hh], dim=1)).chunk(3, dim=1)

        lh_hl_gate, hh_gate = self._gates(spec, ll)
        lh = lh * lh_hl_gate
        hl = hl * lh_hl_gate
        hh = hh * hh_gate

        wave = _haar_idwt2(ll, lh, hl, hh, orig_hw)

        # Only inject the wavelet-induced difference, not a second full feature stream.
        delta = self.post(wave - z)
        return x + self.gamma * delta


# Compatibility aliases for different YAML/parser naming styles.
PseudoSpecularGuidedWaveletSAFR = PGWSAFR
PGW_SAFR = PGWSAFR
