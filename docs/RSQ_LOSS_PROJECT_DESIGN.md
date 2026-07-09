# RSQ-Loss project design for YOLO26-C1

## 1. Project position

This document defines a loss-function improvement project for the water-surface floating waste detection task.

Current main baseline:

```text
YOLO26n + C1 pseudo-specular prior guided feature recalibration
```

New project:

```text
YOLO26n-C1 + RSQ-Loss
```

RSQ-Loss is a training-side improvement. It should not be confused with PG-SAF, SAFSBlock, RAS-v2c, or C1-RGZ.

## 2. Why CIoU is not enough

CIoU is a generic bounding-box loss. It optimizes overlap, center distance, and aspect-ratio consistency. For the FloatingWaste-I water-surface task, this is not fully aligned with the real error sources.

The task has four special properties:

1. **Tiny objects dominate.** Many targets occupy only a few pixels after resizing to 640. A 1-2 px shift can sharply reduce IoU.
2. **Boundary uncertainty is high.** Transparent bottles, wet cartons, light paper, and partially submerged objects often have ambiguous boundaries.
3. **Specular pseudo-edges are strong.** Water reflection, ripple, foam, and shadow edges can attract bounding boxes.
4. **Score quality is unstable.** Some true small objects appear as low-confidence candidates, while high-reflection false positives may have high confidence.

Therefore, the loss should not be only a more complex IoU. It should combine:

```text
small-object tolerant geometry
+ useful-edge alignment
+ specular-core suppression
+ quality-aware score calibration
```

## 3. Proposed method

Name:

```text
RSQ-Loss: Reflectance-guided Small-object Quality Loss
```

Chinese name:

```text
反光引导的小目标质量损失
```

Overall formula:

```text
L_RSQ = λ_box * L_SmallBox
      + λ_edge * α_s * L_SpecEdge
      + λ_core * α_s * L_SpecCore
      + λ_q * L_Quality
      + λ_cls * L_Cls
      + L_DFL if the baseline uses DFL
```

Small-object box term:

```text
L_SmallBox = (1 - α_s) * L_MPDIoU + α_s * L_NWD
```

Small-object weight:

```text
α_s = exp(-sqrt(w_gt * h_gt) / τ)
```

The smaller the target, the larger `α_s` becomes. This makes tiny boxes rely more on the smoother NWD term, while medium/larger boxes rely more on IoU/corner geometry.

## 4. Component details

### 4.1 MPDIoU-style geometry term

Use a corner-distance enhanced IoU term:

```text
L_MPDIoU = 1 - IoU + (d_tl^2 + d_br^2) / (W^2 + H^2 + eps)
```

Where:

- `d_tl`: distance between predicted and target top-left corners;
- `d_br`: distance between predicted and target bottom-right corners;
- `W, H`: width and height of the smallest enclosing box or another stable normalizer;
- `eps`: numerical guard.

Why this fits the task:

- It avoids over-relying on CIoU's aspect-ratio penalty.
- It directly improves corner localization, which matters for AP75.
- It is still compatible with existing YOLO box regression logic.

### 4.2 NWD small-object smoothing term

Approximate the normalized Wasserstein distance between two boxes:

```text
D = sqrt((cx - cx_gt)^2 + (cy - cy_gt)^2 + ((w - w_gt)^2 + (h - h_gt)^2) / 4)
L_NWD = 1 - exp(-D / C)
```

Why this fits the task:

- For tiny boxes, IoU can collapse sharply after only a small pixel shift.
- NWD provides smoother gradients for extremely small objects.
- It can stabilize training for bottle caps, small paper pieces, tiny plastic fragments, and weakly textured targets.

### 4.3 Specular-edge alignment term

C1 already introduces pseudo-specular prior reasoning. RSQ-Loss should use that prior during training if it is available.

Use `Eedge` as a useful edge cue:

```text
L_SpecEdge = I(edge_gt > t_e) * (1 - mean(Eedge on B_pred_boundary))
```

Intuition:

- Transparent bottles and cartons may be visible mainly through edges.
- Reflection edges can be useful, but only when they correspond to target structure.
- This term encourages the predicted box boundary to align with available target/specular edge evidence.

Safety rule:

```text
Do not blindly enhance all high-frequency or high-edge regions.
```

Waves, foam, and shadows also contain edges. This term should be gated by target-region evidence and positive sample assignment.

### 4.4 Specular-core suppression term

Use `Rcore` as a high-risk specular core map.

Penalize predicted area outside the target that expands into specular core:

```text
L_SpecCore = mean(Rcore inside (B_pred \ B_gt))
```

Approximation:

```text
L_SpecCore = mean(pred_mask * (1 - gt_mask) * Rcore)
```

Intuition:

- Do not punish true object pixels just because they are near reflection.
- Punish the part of the predicted box that leaks into strong reflection core outside the GT.
- This is safer than simply down-weighting all high-reflection areas.

### 4.5 Quality-aware score calibration term

Define a detached target quality:

```text
q_star = IoU_detach^gamma * Q_spec
Q_spec = clamp(1 + a * E_box - b * R_extra, 0, 1)
```

Where:

- `IoU_detach`: detached IoU between predicted and target box;
- `E_box`: useful edge evidence near predicted boundary;
- `R_extra`: specular-core risk in the predicted area outside GT;
- `gamma`: quality sharpness;
- `a`: edge reward coefficient;
- `b`: core penalty coefficient.

Use this quality target to improve classification/score calibration without adding a new Detect output whenever possible.

Goal:

```text
true small object + good edge alignment + low extra specular core -> higher score
reflection pseudo target + poor localization + high extra specular core -> lower score
```

## 5. Recommended hyperparameters

Initial values:

```yaml
box_loss_type: rsq
rsq_tau: 32.0
rsq_nwd_c: 16.0
rsq_lambda_box: 7.5
rsq_lambda_edge: 0.05
rsq_lambda_core: 0.05
rsq_lambda_quality: 0.3
rsq_edge_thr: 0.25
rsq_quality_gamma: 1.0
rsq_quality_edge_gain: 0.2
rsq_quality_core_penalty: 0.3
rsq_warmup_epochs: 10
rsq_enable_spec_edge: true
rsq_enable_spec_core: true
rsq_enable_quality: false
```

Warm-up recommendation:

```text
Epoch 0-10:
  use only MPDIoU + NWD + original cls/DFL terms

After epoch 10:
  gradually enable SpecEdge and SpecCore

After RSQ-v1 is stable:
  enable Quality term as RSQ-v2
```

## 6. Implementation architecture

Recommended file layout in the actual YOLO26 codebase:

```text
ultralytics/utils/rsq_loss.py
ultralytics/utils/loss.py
configs/hyp-rsq-loss.yaml
configs/yolo26n-c1-rsq.yaml
scripts/train_c1_rsq_loss.sh or scripts/train_c1_rsq_loss.md
```

### 6.1 `rsq_loss.py`

Suggested helper functions:

```python
bbox_iou_xyxy(pred, target, eps=1e-7)
mpdiou_loss_xyxy(pred, target, eps=1e-7)
nwd_loss_xyxy(pred, target, c=16.0, eps=1e-7)
small_object_weight_xyxy(target, tau=32.0, eps=1e-7)
box_boundary_ring_mask(...)
spec_edge_loss(...)
spec_core_loss(...)
rsq_quality_target(...)
```

Keep helpers independently testable.

### 6.2 `loss.py`

Expected integration:

- keep original CIoU path as default;
- add `box_loss_type` switch;
- call MPDIoU/NWD/RSQ only when explicitly enabled;
- preserve original DFL behavior if DFL exists;
- preserve original positive sample normalization.

### 6.3 Specular map wiring

Preferred order:

1. Reuse existing C1 prior maps if already available in batch/model outputs.
2. If only `Pspec` exists, derive approximate `Rcore` and `Eedge` from it non-invasively.
3. If prior maps cannot reach loss without major model changes, first run Stage A box-only loss and document that RSQ-v1/v2 requires prior-map exposure.

Do not rewrite C1 just to pass prior maps.

## 7. Experiment plan

Run loss variants in increasing complexity:

| ID | Method | Purpose |
|---|---|---|
| L0 | C1 original loss | baseline |
| L1 | C1 + MPDIoU | remove CIoU aspect-ratio risk |
| L2 | C1 + NWD | test small-object smoothing |
| L3 | C1 + MPDIoU + NWD | box-only small-object hybrid |
| L4 | C1 + RSQ-v1 | add SpecEdge and SpecCore |
| L5 | C1 + RSQ-v2 | add quality-aware score calibration |

Evaluation metrics:

```text
AP
AP50
AP75
APs
ARs
reflection_highlight_background FP
wave/ripple/foam FP
small FN
localization_partial_overlap
duplicate boxes
latency
```

## 8. Acceptance criteria

A result is promising if:

- AP75 improves;
- APs improves;
- ARs improves or remains stable;
- reflection false positives decrease or do not increase;
- AP50 does not collapse;
- bottle and carton both remain stable;
- training is stable without NaN/Inf.

A result should be rejected if:

- AP50 drops by more than 1.0;
- APs drops;
- reflection/wave false positives increase while AP gain is tiny;
- improvement appears only in AP but not AP75/APs;
- specular map computation introduces large runtime or memory overhead.

## 9. Paper narrative

Weak narrative:

```text
We replace CIoU with a better IoU loss.
```

Recommended narrative:

```text
We propose RSQ-Loss, a reflectance-guided small-object quality loss for water-surface floating waste detection. It combines small-object tolerant box regression, pseudo-specular edge alignment, specular-core suppression, and quality-aware score calibration to address boundary ambiguity and reflection-induced false localization.
```

Chinese version:

```text
针对水面漂浮物检测中小目标边界模糊、镜面高光伪边界强、分类分数与定位质量不一致的问题，本文提出反光引导的小目标质量损失 RSQ-Loss。该损失在几何定位项中引入小目标平滑距离约束，并利用伪镜面先验构造边缘一致性项与高光核心规避项，同时通过质量感知分类目标提升候选排序可靠性。
```

## 10. Relationship to RAS/RGZ

RSQ-Loss does not replace RAS/RGZ.

RAS/RGZ solves:

```text
image-space pixel visibility bottleneck
```

RSQ-Loss solves:

```text
training-side localization quality + reflection-aware score calibration
```

Expected role:

```text
C1 full-image baseline + RSQ-Loss -> stronger single-forward model
C1-RGZ or RAS selective inference -> image-space zoom upper-bound direction
```

If RSQ-Loss improves AP75/APs without increasing false positives, it can become the training-side companion to C1 and a strong baseline before RGZ.
