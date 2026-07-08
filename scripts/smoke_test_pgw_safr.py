"""Smoke test for PGW-SAFR.

Run from repository root:
    python scripts/smoke_test_pgw_safr.py

This does not train a detector. It only verifies:
    - import works;
    - odd/even feature sizes are preserved;
    - optional spec prior can be passed;
    - gamma=0 initialization makes the module an exact residual no-op.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from modules.pgw_safr import PGWSAFR


def _check_case(batch: int, channels: int, height: int, width: int, with_spec: bool) -> None:
    torch.manual_seed(0)
    module = PGWSAFR(channels, mode="safr")
    x = torch.randn(batch, channels, height, width)
    if with_spec:
        spec = torch.rand(batch, 2, height * 8, width * 8)
        y = module([x, spec])
    else:
        y = module(x)

    assert y.shape == x.shape, f"shape mismatch: got {tuple(y.shape)}, expected {tuple(x.shape)}"
    max_abs = (y - x).abs().max().item()
    assert max_abs == 0.0, f"gamma=0 should be exact no-op, max_abs={max_abs}"


if __name__ == "__main__":
    _check_case(batch=2, channels=64, height=80, width=80, with_spec=True)
    _check_case(batch=1, channels=64, height=81, width=79, with_spec=False)
    _check_case(batch=1, channels=128, height=40, width=40, with_spec=True)
    print("PGW-SAFR smoke test passed.")
