from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.zoom_view_correction import (
    CropMeta,
    ZVCConfig,
    map_boxes_crop_to_full,
    map_boxes_full_to_crop,
    make_square_crop_around_box,
    positive_zoom_distillation_loss,
    reflective_hard_negative_loss,
)


def test_box_mapping_round_trip() -> None:
    meta = CropMeta(
        image_index=0,
        crop_xyxy_full=(100.0, 50.0, 300.0, 250.0),
        crop_hw=(640, 640),
        full_hw=(480, 640),
    )
    full = torch.tensor([[120.0, 70.0, 180.0, 150.0]])
    crop = map_boxes_full_to_crop(full, meta)
    restored = map_boxes_crop_to_full(crop, meta)
    assert torch.allclose(full, restored, atol=1e-5)


def test_positive_distillation_is_finite_and_backpropagates() -> None:
    cfg = ZVCConfig(
        warmup_epochs=0,
        ramp_epochs=0,
        teacher_pos_conf=0.2,
        teacher_pos_iou=0.4,
        match_iou_min=0.05,
    )
    meta = CropMeta(
        image_index=0,
        crop_xyxy_full=(100.0, 100.0, 300.0, 300.0),
        crop_hw=(640, 640),
        full_hw=(640, 640),
    )

    # Teacher crop box maps to [125, 125, 175, 175] in the full image.
    teacher_boxes_crop = torch.tensor([[80.0, 80.0, 240.0, 240.0]])
    gt_boxes_crop = teacher_boxes_crop.clone()
    gt_labels = torch.tensor([1])
    teacher_logits = torch.tensor([[-4.0, 4.0]])

    student_boxes = torch.tensor(
        [[123.0, 126.0, 177.0, 176.0], [400.0, 400.0, 450.0, 450.0]],
        requires_grad=True,
    )
    student_logits = torch.tensor(
        [[-1.0, 0.5], [2.0, -2.0]],
        requires_grad=True,
    )

    loss, logs = positive_zoom_distillation_loss(
        student_boxes_full=student_boxes,
        student_logits=student_logits,
        teacher_boxes_crop=teacher_boxes_crop,
        teacher_logits=teacher_logits,
        gt_boxes_crop=gt_boxes_crop,
        gt_labels=gt_labels,
        meta=meta,
        cfg=cfg,
        epoch=1,
    )
    assert torch.isfinite(loss)
    assert logs["zvc_pos_pairs"].item() == 1
    loss.backward()
    assert student_boxes.grad is not None
    assert student_logits.grad is not None


def test_positive_distillation_rejects_unreliable_teacher() -> None:
    cfg = ZVCConfig(warmup_epochs=0, ramp_epochs=0, teacher_pos_conf=0.9)
    meta = CropMeta(0, (0.0, 0.0, 320.0, 320.0), (640, 640), (640, 640))
    student_boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]], requires_grad=True)
    student_logits = torch.zeros((1, 2), requires_grad=True)
    teacher_boxes = torch.tensor([[20.0, 20.0, 100.0, 100.0]])
    teacher_logits = torch.zeros((1, 2))
    gt_boxes = teacher_boxes.clone()
    gt_labels = torch.tensor([0])

    loss, logs = positive_zoom_distillation_loss(
        student_boxes,
        student_logits,
        teacher_boxes,
        teacher_logits,
        gt_boxes,
        gt_labels,
        meta,
        cfg,
        epoch=1,
    )
    assert loss.item() == 0.0
    assert logs["zvc_pos_pairs"].item() == 0.0


def test_hard_negative_requires_verified_empty_crop() -> None:
    cfg = ZVCConfig(
        warmup_epochs=0,
        ramp_epochs=0,
        teacher_neg_conf_max=0.1,
        student_hardneg_conf=0.1,
    )
    meta = CropMeta(0, (0.0, 0.0, 200.0, 200.0), (640, 640), (640, 640))
    student_boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    student_logits = torch.tensor([[2.0, -2.0]], requires_grad=True)

    blocked, blocked_logs = reflective_hard_negative_loss(
        student_boxes,
        student_logits,
        meta,
        crop_has_gt=True,
        teacher_max_conf=0.0,
        reflection_reliability=1.0,
        cfg=cfg,
        epoch=1,
    )
    assert blocked.item() == 0.0
    assert blocked_logs["zvc_hardneg_count"].item() == 0.0

    active, active_logs = reflective_hard_negative_loss(
        student_boxes,
        student_logits,
        meta,
        crop_has_gt=False,
        teacher_max_conf=0.0,
        reflection_reliability=1.0,
        cfg=cfg,
        epoch=1,
    )
    assert active.item() > 0.0
    assert active_logs["zvc_hardneg_count"].item() == 1.0


def test_square_crop_contains_target_with_zero_jitter() -> None:
    generator = torch.Generator().manual_seed(0)
    crop = make_square_crop_around_box(
        torch.tensor([20.0, 30.0, 40.0, 50.0]),
        (100, 120),
        context_scale=3.0,
        min_crop_size=32.0,
        center_jitter=0.0,
        scale_jitter=0.0,
        generator=generator,
    )
    x1, y1, x2, y2 = crop
    assert x1 <= 20.0 <= x2
    assert y1 <= 30.0 <= y2
    assert x1 <= 40.0 <= x2
    assert y1 <= 50.0 <= y2
