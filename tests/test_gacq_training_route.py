from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.gacq_training_route import (
    GACQAuxiliaryHead,
    GACQConfig,
    auxiliary_ramp,
    build_gt_candidate_points,
    gacq_auxiliary_loss,
    gather_auxiliary_candidates,
    strip_auxiliary_state_dict,
    teacher_reliability_weight,
)


def test_reliability_uses_geometric_mean_and_rejects_bad_teacher() -> None:
    cfg = GACQConfig()
    conf = torch.tensor([0.8, 0.2])
    iou = torch.tensor([0.7, 0.9])
    visible = torch.tensor([1.0, 1.0])
    weight, valid = teacher_reliability_weight(conf, iou, visible, None, None, cfg)
    assert valid.tolist() == [True, False]
    assert 0.0 < weight[0].item() <= 1.0
    assert weight[1].item() == 0.0
    assert not weight.requires_grad


def test_fixed_gt_points_are_center_first() -> None:
    cfg = GACQConfig(candidate_radius=1, max_candidates_per_gt=5)
    gt = torch.tensor([[16.0, 16.0, 32.0, 32.0]])
    points = build_gt_candidate_points(gt, [(80, 80), (40, 40), (20, 20)], cfg)
    assert len(points) == 5
    assert points[0].level_index == 0
    assert points[0].x == 3 and points[0].y == 3


def test_auxiliary_loss_backpropagates_to_non_detached_shared_features() -> None:
    torch.manual_seed(0)
    cfg = GACQConfig(warmup_epochs=0, ramp_epochs=0, max_candidates_per_gt=5)
    features = [
        torch.randn(1, 16, 8, 8, requires_grad=True),
        torch.randn(1, 24, 4, 4, requires_grad=True),
        torch.randn(1, 32, 2, 2, requires_grad=True),
    ]
    head = GACQAuxiliaryHead([16, 24, 32], num_classes=2, hidden_channels=8)
    outputs = head(features)
    gt_boxes = torch.tensor([[12.0, 12.0, 28.0, 28.0]])
    gt_labels = torch.tensor([1], dtype=torch.long)
    points = build_gt_candidate_points(gt_boxes, [(8, 8), (4, 4), (2, 2)], cfg)
    gathered = gather_auxiliary_candidates(outputs, points, image_index=0)
    loss, logs = gacq_auxiliary_loss(
        gathered, gt_boxes, gt_labels, torch.tensor([0.9]), cfg, epoch=1
    )
    assert torch.isfinite(loss)
    assert logs["gacq_candidates"].item() == 5
    loss.backward()
    assert features[0].grad is not None
    assert torch.isfinite(features[0].grad).all()
    assert features[0].grad.abs().sum().item() > 0.0
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in head.parameters())


def test_zero_reliability_produces_finite_zero_weighted_loss() -> None:
    torch.manual_seed(1)
    cfg = GACQConfig(warmup_epochs=0, ramp_epochs=0)
    features = [torch.randn(1, 8, 8, 8), torch.randn(1, 8, 4, 4), torch.randn(1, 8, 2, 2)]
    head = GACQAuxiliaryHead([8, 8, 8], num_classes=2, hidden_channels=8)
    outputs = head(features)
    gt_boxes = torch.tensor([[12.0, 12.0, 28.0, 28.0]])
    gt_labels = torch.tensor([0], dtype=torch.long)
    points = build_gt_candidate_points(gt_boxes, [(8, 8), (4, 4), (2, 2)], cfg)
    gathered = gather_auxiliary_candidates(outputs, points, image_index=0)
    loss, _ = gacq_auxiliary_loss(
        gathered, gt_boxes, gt_labels, torch.tensor([0.0]), cfg, epoch=1
    )
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_warmup_and_ramp() -> None:
    cfg = GACQConfig(warmup_epochs=2, ramp_epochs=4)
    assert auxiliary_ramp(0, cfg) == 0.0
    assert auxiliary_ramp(1, cfg) == 0.0
    assert auxiliary_ramp(2, cfg) == 0.25
    assert auxiliary_ramp(5, cfg) == 1.0


def test_strip_auxiliary_state_dict() -> None:
    state = {
        "model.backbone.weight": torch.tensor(1.0),
        "gacq_aux.head.weight": torch.tensor(2.0),
        "crop_teacher.model.weight": torch.tensor(3.0),
    }
    stripped = strip_auxiliary_state_dict(state)
    assert list(stripped) == ["model.backbone.weight"]
