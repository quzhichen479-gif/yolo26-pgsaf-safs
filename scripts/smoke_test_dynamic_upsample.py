"""Smoke test for DySample / CARAFE / CARAFEPlusLite / C1P4P3DynamicFusion.

Run from repository root:
    python scripts/smoke_test_dynamic_upsample.py

This does not train a detector. It verifies:
    - imports work;
    - P4-like feature maps upsample to P3-like shape;
    - residual fusion wrapper keeps gamma=0 as an exact no-op;
    - optional spec prior can be passed.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from modules.dynamic_upsample import CARAFE, CARAFEPlusLite, C1P4P3DynamicFusion, DySample


def _check_upsampler() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 16, 8, 8)
    ref = torch.randn(1, 16, 16, 16)

    modules = [
        DySample(16, scale=2, groups=4),
        CARAFE(16, scale=2, compress_channels=8),
        CARAFEPlusLite(16, scale=2, compress_channels=8),
    ]
    for module in modules:
        y = module([x, ref])
        assert y.shape == ref.shape, f"{type(module).__name__}: got {tuple(y.shape)}, expected {tuple(ref.shape)}"


def _check_fusion() -> None:
    torch.manual_seed(0)
    p3 = torch.randn(1, 16, 16, 16)
    p4 = torch.randn(1, 32, 8, 8)
    spec = torch.rand(1, 2, 64, 64)

    for upsampler in ["dysample", "carafe", "carafe_plus"]:
        module = C1P4P3DynamicFusion(16, 32, upsampler=upsampler, use_spec_gate=True)
        y = module([p3, p4, spec])
        assert y.shape == p3.shape, f"{upsampler}: got {tuple(y.shape)}, expected {tuple(p3.shape)}"
        max_abs = (y - p3).abs().max().item()
        assert max_abs == 0.0, f"{upsampler}: gamma=0 should be exact no-op, max_abs={max_abs}"


if __name__ == "__main__":
    _check_upsampler()
    _check_fusion()
    print("Dynamic upsample smoke test passed.")
