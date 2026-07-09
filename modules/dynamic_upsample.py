"""
Dynamic upsampling modules for YOLO26-C1 P4-to-P3 ablations.

Implemented modules:
    - DySample: lightweight dynamic point-sampling upsampler.
    - CARAFE: dependency-free content-aware reassembly upsampler.
    - CARAFEPlusLite: CARAFE++-style stronger content-aware upsampler.
    - C1P4P3DynamicFusion: residual-safe P4-to-P3 semantic injection wrapper.

These modules are intended for controlled C1 experiments:
    - full-image one-pass inference only;
    - no Detect/loss/assignment changes;
    - first replace or wrap the P4->P3 upsampling/fusion path only.
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


def _resize_like(x: torch.Tensor, ref: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    if mode in {"nearest", "area"}:
        return F.interpolate(x, size=ref.shape[-2:], mode=mode)
    return F.interpolate(x, size=ref.shape[-2:], mode=mode, align_corners=False)


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


def _init_weights(module: nn.Module, std: float = 0.001) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class DySample(nn.Module):
    """Dynamic sampling upsampler.

    This is a dependency-free implementation of the common DySample-style idea:
    learn a small offset field and upsample by grid_sample rather than fixed
    nearest/bilinear interpolation.

    Args:
        c: input channels.
        scale: upsampling factor. First C1 ablation should use scale=2 for P4->P3.
        groups: channel groups for sampling. c must be divisible by groups.
        use_scope: if True, learn a dynamic scope to bound offsets.
        offset_scale: conservative multiplier for learned offsets.
    """

    def __init__(self, c: int, scale: int = 2, groups: int = 4, use_scope: bool = False, offset_scale: float = 0.25):
        super().__init__()
        if c % groups != 0:
            raise ValueError(f"DySample requires c % groups == 0, got c={c}, groups={groups}")
        self.c = c
        self.scale = int(scale)
        self.groups = int(groups)
        self.offset_scale = float(offset_scale)

        self.offset = nn.Conv2d(c, 2 * groups * self.scale * self.scale, kernel_size=1)
        _init_weights(self.offset, std=0.001)

        self.scope = None
        if use_scope:
            self.scope = nn.Conv2d(c, 2 * groups * self.scale * self.scale, kernel_size=1)
            nn.init.constant_(self.scope.weight, 0.0)
            nn.init.constant_(self.scope.bias, 0.0)

        self.register_buffer("init_pos", self._init_pos(), persistent=False)

    def _init_pos(self) -> torch.Tensor:
        # Initial sub-pixel offsets around each low-resolution cell.
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1, dtype=torch.float32) / self.scale
        yy, xx = torch.meshgrid(h, h, indexing="ij")
        base = torch.stack([xx, yy], dim=0).reshape(2, -1)  # [2, scale^2]
        base = base.repeat(1, self.groups).reshape(1, 2 * self.groups * self.scale * self.scale, 1, 1)
        return base

    def _sample(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        b, _, h, w = offset.shape
        s = self.scale
        g = self.groups

        # [B, 2*g*s*s, H, W] -> [B, 2, g*s*s, H, W]
        offset = offset.view(b, 2, g * s * s, h, w)

        yy = torch.arange(h, dtype=x.dtype, device=x.device) + 0.5
        xx = torch.arange(w, dtype=x.dtype, device=x.device) + 0.5
        yy, xx = torch.meshgrid(yy, xx, indexing="ij")
        coords = torch.stack([xx, yy], dim=0).view(1, 2, 1, h, w)
        coords = coords + offset

        # Normalize to grid_sample coordinates at low resolution.
        normalizer = x.new_tensor([w, h]).view(1, 2, 1, 1, 1)
        coords = 2.0 * coords / normalizer - 1.0

        # Rearrange sub-pixel coordinates to high-resolution grid.
        coords = coords.view(b, 2 * g * s * s, h, w)
        coords = F.pixel_shuffle(coords, s)  # [B, 2*g, H*s, W*s]
        coords = coords.view(b, 2, g, h * s, w * s)
        coords = coords.permute(0, 2, 3, 4, 1).contiguous()  # [B,g,Hs,Ws,2]
        coords = coords.view(b * g, h * s, w * s, 2)

        xg = x.view(b * g, self.c // g, h, w)
        out = F.grid_sample(xg, coords, mode="bilinear", align_corners=False, padding_mode="border")
        return out.view(b, self.c, h * s, w * s)

    def forward(self, inputs: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            x, ref = inputs, None
        else:
            if len(inputs) < 1:
                raise ValueError("DySample expects x or [x, optional_ref]")
            x = inputs[0]
            ref = inputs[1] if len(inputs) > 1 else None

        offset = self.offset(x)
        if self.scope is not None:
            offset = offset * self.scope(x).sigmoid()
        offset = self.init_pos + offset * self.offset_scale

        y = self._sample(x, offset)
        if ref is not None:
            y = _resize_like(y, ref)
        return y


class CARAFE(nn.Module):
    """Content-Aware ReAssembly of FEatures upsampler.

    Dependency-free CARAFE-style implementation using unfold + softmax masks.
    It is slower than optimized CUDA implementations but suitable as a Codex-ready
    controlled ablation and smoke-testable PyTorch prototype.
    """

    def __init__(
        self,
        c: int,
        scale: int = 2,
        compress_channels: int = 64,
        encoder_kernel: int = 3,
        up_kernel: int = 5,
    ):
        super().__init__()
        self.c = c
        self.scale = int(scale)
        self.up_kernel = int(up_kernel)
        self.comp = ConvBNAct(c, compress_channels, k=1)
        self.encoder = nn.Conv2d(
            compress_channels,
            (self.scale * self.scale) * (self.up_kernel * self.up_kernel),
            kernel_size=encoder_kernel,
            padding=encoder_kernel // 2,
        )
        nn.init.normal_(self.encoder.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.encoder.bias, 0.0)

    def _reassemble(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        s = self.scale
        k = self.up_kernel
        hs, ws = h * s, w * s

        # mask: [B, s*s*k*k, H, W] -> [B, k*k, H*s, W*s]
        mask = F.pixel_shuffle(mask, s)
        mask = mask.view(b, k * k, hs, ws)
        mask = F.softmax(mask, dim=1)

        # Reassemble nearest-upsampled features with content-aware kernels.
        x_up = F.interpolate(x, scale_factor=s, mode="nearest")
        patches = F.unfold(x_up, kernel_size=k, padding=k // 2)
        patches = patches.view(b, c, k * k, hs, ws)
        out = (patches * mask.unsqueeze(1)).sum(dim=2)
        return out

    def forward(self, inputs: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            x, ref = inputs, None
        else:
            if len(inputs) < 1:
                raise ValueError("CARAFE expects x or [x, optional_ref]")
            x = inputs[0]
            ref = inputs[1] if len(inputs) > 1 else None

        mask = self.encoder(self.comp(x))
        y = self._reassemble(x, mask)
        if ref is not None:
            y = _resize_like(y, ref)
        return y


class CARAFEPlusLite(nn.Module):
    """CARAFE++-style stronger content-aware upsampler.

    This is not a drop-in copy of the official optimized CARAFE++ operator.
    It keeps the same controlled-ablation spirit: larger context for mask
    generation plus light post-reassembly refinement.
    """

    def __init__(
        self,
        c: int,
        scale: int = 2,
        compress_channels: int = 64,
        encoder_kernel: int = 5,
        up_kernel: int = 5,
    ):
        super().__init__()
        self.carafe = CARAFE(c, scale, compress_channels, encoder_kernel, up_kernel)
        self.post = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, inputs: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
        y = self.carafe(inputs)
        return y + self.gamma * self.post(y)


class C1P4P3DynamicFusion(nn.Module):
    """Residual-safe C1 P4-to-P3 semantic injection wrapper.

    Intended usage:
        P3_refined = C1P4P3DynamicFusion(c3, c4, upsampler="dysample")([P3_C1, P4_C1, spec])

    It does not replace Detect. It only injects dynamically upsampled P4 context
    into C1's P3 feature with gamma=0 residual initialization.
    """

    def __init__(
        self,
        c3: int,
        c4: int,
        c_out: Optional[int] = None,
        upsampler: str = "dysample",
        reduction: float = 0.5,
        groups: int = 4,
        use_spec_gate: bool = False,
    ):
        super().__init__()
        c_out = c_out or c3
        c_mid = max(16, int(c_out * reduction))
        self.use_spec_gate = bool(use_spec_gate)

        self.p3_proj = ConvBNAct(c3, c_mid, k=1)
        self.p4_proj = ConvBNAct(c4, c_mid, k=1)

        upsampler = upsampler.lower()
        if upsampler == "dysample":
            valid_groups = min(groups, c_mid)
            while c_mid % valid_groups != 0 and valid_groups > 1:
                valid_groups -= 1
            self.up = DySample(c_mid, scale=2, groups=valid_groups)
        elif upsampler == "carafe":
            self.up = CARAFE(c_mid, scale=2, compress_channels=max(16, c_mid // 2))
        elif upsampler in {"carafe_plus", "carafe++", "carafeplus"}:
            self.up = CARAFEPlusLite(c_mid, scale=2, compress_channels=max(16, c_mid // 2))
        else:
            raise ValueError(f"unknown upsampler: {upsampler}")

        self.align_gate = nn.Sequential(
            ConvBNAct(2 * c_mid, c_mid, k=3),
            nn.Conv2d(c_mid, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fuse = ConvBNAct(2 * c_mid, c_out, k=3)
        self.shortcut = ConvBNAct(c3, c_out, k=1) if c3 != c_out else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.spec_ab = nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float32))

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor) or len(inputs) < 2:
            raise ValueError("C1P4P3DynamicFusion expects [P3_C1, P4_C1, optional_spec]")
        p3, p4 = inputs[0], inputs[1]
        spec = inputs[2] if len(inputs) > 2 else None

        f3 = self.p3_proj(p3)
        f4 = self.p4_proj(p4)
        f4 = self.up([f4, f3])

        content_gate = self.align_gate(torch.cat([f3, f4], dim=1))
        if self.use_spec_gate:
            rcore, eedge = _split_spec(spec, f3)
            edge_gain = torch.relu(self.spec_ab[0])
            core_penalty = torch.relu(self.spec_ab[1])
            spec_gate = torch.sigmoid(edge_gain * eedge - core_penalty * rcore)
            gate = content_gate * spec_gate
        else:
            gate = content_gate

        fused = self.fuse(torch.cat([f3, f4 * gate], dim=1))
        return self.shortcut(p3) + self.gamma * fused


# Compatibility aliases for different YAML/parser naming styles.
DynamicP4P3Fusion = C1P4P3DynamicFusion
CARAFEPlus = CARAFEPlusLite
