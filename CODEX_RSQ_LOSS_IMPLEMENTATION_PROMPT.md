# Codex implementation prompt: YOLO26-C1 + RSQ-Loss

You are working in the existing YOLO26 water-surface floating waste detection repository. Implement a controlled loss-function ablation for the existing C1 baseline.

This task is **not** a new neck module, not a new detection head, and not a new RAS/RGZ inference pipeline. It is a training-side loss improvement designed for the FloatingWaste-I water-surface small-object detection task.

## 0. Mandatory reading before editing code

Before writing code, read and summarize the following files in your own implementation notes:

### A. Project-level instruction files

1. `README.md`
2. `CODEX_IMPLEMENTATION_PROMPT.md`
3. `CODEX_RSQ_LOSS_IMPLEMENTATION_PROMPT.md`  ← this file
4. `docs/RSQ_LOSS_PROJECT_DESIGN.md` if present
5. `scripts/run_rsq_loss_experiment_matrix.md` if present

### B. Existing YOLO26-C1 model files

Find and read the current C1 configuration and C1 module implementation. Use local search commands such as:

```bash
find . -iname "*c1*.yaml" -o -iname "*safrg*.yaml" -o -iname "*spec*.yaml"
grep -R "SpecularAwareFeatureRecalibration\|SAFRG\|Pspec\|specular\|Rcore\|Eedge" -n .
```

You must identify:

- the current C1 YAML/config file;
- where `SpecularAwareFeatureRecalibration` or the equivalent C1 recalibration module is defined;
- whether C1 already exposes a pseudo-specular prior, `Rcore`, `Eedge`, `Pspec`, or a similar map;
- whether that prior is available inside the loss path or only inside the model forward path.

### C. Existing loss / assignment / training files

Find and read the current loss implementation before modifying anything. Typical locations may include:

```bash
find . -path "*loss*.py" -o -path "*tal*.py" -o -path "*assigner*.py" -o -path "*train*.py"
grep -R "CIoU\|bbox_loss\|BboxLoss\|v8DetectionLoss\|TaskAlignedAssigner\|DFL" -n ultralytics .
```

Read these files if they exist:

- `ultralytics/utils/loss.py`
- `ultralytics/utils/tal.py`
- `ultralytics/models/yolo/detect/train.py`
- `ultralytics/cfg/default.yaml` or the repository's equivalent hyperparameter config
- any custom YOLO26 loss file if the repository has renamed paths

Important: `tal.py` / assignment code should be read for context, but **do not modify assignment** unless absolutely necessary and explicitly documented.

## 1. Background and motivation

The current validated C1 baseline is a stable full-image detector using pseudo-specular prior guided feature recalibration. It is good at suppressing water-surface reflection noise, but the task still suffers from:

- extremely small objects whose short side can be only a few pixels at 640 input;
- weak texture from transparent bottles, wet cartons, and light-colored paper;
- strong water reflection, ripple, foam, and shadow pseudo-edges;
- AP75 sensitivity caused by small boundary errors;
- low-confidence true positives and score/localization mismatch.

CIoU is too generic for this situation. It does not distinguish useful object/specular edges from harmful specular-core pseudo boundaries, and its aspect-ratio term can be noisy for tiny or partially occluded floating objects.

## 2. Objective

Implement **RSQ-Loss: Reflectance-guided Small-object Quality Loss** as an opt-in loss for C1.

Full name:

```text
RSQ-Loss = Reflectance-guided Small-object Quality Loss
```

Chinese name:

```text
反光引导的小目标质量损失
```

The loss should improve small-object localization and score quality without changing:

- model backbone;
- C1 module structure;
- Detect head;
- assignment strategy;
- dataset split;
- evaluation protocol.

## 3. High-level formula

Implement the loss in staged form:

```text
L_RSQ = λ_box * L_SmallBox
      + λ_edge * α_s * L_SpecEdge
      + λ_core * α_s * L_SpecCore
      + λ_q * L_Quality
      + λ_cls * L_Cls
      + existing DFL term if the baseline uses DFL
```

Where:

```text
L_SmallBox = (1 - α_s) * L_MPDIoU + α_s * L_NWD
```

and:

```text
α_s = exp(-sqrt(w_gt * h_gt) / τ)
```

Use box sizes in input-image pixel scale whenever possible. If the current loss operates in feature/grid scale, convert consistently or document the chosen scale.

## 4. Components to implement

### 4.1 Small-object box loss

Replace or wrap the current CIoU box regression term with a small-object-aware hybrid:

```text
L_SmallBox = (1 - α_s) * L_MPDIoU + α_s * L_NWD
```

#### MPDIoU-style term

Use a corner-distance IoU term:

```text
L_MPDIoU = 1 - IoU + (d_tl^2 + d_br^2) / (W^2 + H^2 + eps)
```

Where:

- `d_tl` is the distance between predicted and target top-left corners;
- `d_br` is the distance between predicted and target bottom-right corners;
- `W, H` are the width and height of the smallest enclosing rectangle or a stable image-scale normalization;
- `eps` prevents division by zero.

If the repository already contains MPDIoU or corner-distance IoU, reuse it rather than duplicating code.

#### NWD-style term

Implement normalized Wasserstein distance approximation:

```text
D = sqrt((cx - cx_gt)^2 + (cy - cy_gt)^2 + ((w - w_gt)^2 + (h - h_gt)^2) / 4)
L_NWD = 1 - exp(-D / C)
```

Initial recommended `C`:

```text
C = 12.8 or 16.0
```

Expose it as a hyperparameter.

### 4.2 Specular-edge alignment term

If an `Eedge` map is available from C1 or can be derived non-invasively from C1's pseudo-specular prior, implement:

```text
L_SpecEdge = 1 - mean(Eedge on predicted box boundary)
```

Apply it only to positive samples and preferably only when the target region has enough edge evidence:

```text
L_SpecEdge = I(edge_gt > t_e) * (1 - mean(Eedge on B_pred_boundary))
```

Initial threshold:

```text
t_e = 0.2 or 0.3
```

Implementation notes:

- Downsample or sample `Eedge` to the image/loss coordinate system consistently.
- Boundary sampling can be approximate: use a thin rectangular ring around the predicted box.
- If direct differentiable sampling is too complex at first, implement a safe approximate version and document it.
- Do not introduce expensive image preprocessing if the current C1 prior already provides usable specular maps.

### 4.3 Specular-core suppression term

If an `Rcore` map is available, penalize only the predicted area outside the target box that overlaps high-risk specular core:

```text
L_SpecCore = mean(Rcore inside (B_pred \ B_gt))
```

Important behavior:

- Do **not** simply penalize all high-reflection pixels inside the GT box.
- Transparent bottles and wet cartons may overlap useful reflection edges.
- Penalize the extra predicted region that expands into specular core.

If exact set subtraction is difficult, approximate it with masks:

```text
pred_mask * (1 - gt_mask) * Rcore
```

For efficiency, this can be computed on a downsampled prior map.

### 4.4 Quality-aware score calibration term

Implement this as the second stage after the box-only version is stable.

Define a detached quality target:

```text
q_star = IoU_detach^gamma * Q_spec
Q_spec = clamp(1 + a * E_box - b * R_extra, 0, 1)
```

Where:

- `IoU_detach` is detached IoU between predicted and target box;
- `E_box` is boundary/box edge evidence from `Eedge`;
- `R_extra` is extra predicted specular-core risk from `Rcore` outside GT;
- `gamma` default: 1.0 or 1.5;
- `a` default: 0.2;
- `b` default: 0.3.

Then train either:

1. the existing classification score using BCE/Varifocal-style target `q_star`; or
2. a separate quality scalar if the current codebase already supports such a branch.

Prefer option 1 if it can be done without changing Detect. Do not add a new Detect output unless explicitly necessary.

## 5. Hyperparameters

Add safe, opt-in hyperparameters. Suggested names:

```yaml
box_loss_type: ciou          # ciou | mpdiou_nwd | rsq
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

Use the repository's existing hyperparameter style. If names must be shortened, keep them explicit.

Default behavior must remain the original loss unless `box_loss_type` or an equivalent flag enables RSQ.

## 6. Required implementation stages

### Stage A: non-invasive box-loss ablation

Implement:

```text
C1 + MPDIoU
C1 + NWD
C1 + MPDIoU + NWD
```

Do not use specular maps yet.

Acceptance for Stage A:

- forward smoke test passes;
- loss is finite;
- no NaN/Inf in first 100 iterations;
- AP50 does not collapse;
- APs or AP75 improves compared with C1+CIoU.

### Stage B: RSQ-v1 with specular prior

Implement:

```text
L_RSQ-v1 = L_SmallBox + λ_edge * α_s * L_SpecEdge + λ_core * α_s * L_SpecCore
```

Acceptance for Stage B:

- AP does not drop obviously;
- AP75 or APs improves;
- reflection_highlight_background FP decreases or does not increase;
- wave/ripple/foam false positives do not increase obviously.

### Stage C: RSQ-v2 with quality calibration

Implement:

```text
L_RSQ-v2 = L_RSQ-v1 + λ_q * L_Quality
```

Acceptance for Stage C:

- AP75 improves;
- low-confidence true positives are reduced;
- duplicate/partial-overlap predictions decrease;
- precision improves or remains stable while APs improves.

## 7. Required files to create or modify

Because repository layouts vary, adapt the paths to the actual YOLO26 codebase. Expected changes:

### Create if appropriate

- `ultralytics/utils/rsq_loss.py` or equivalent helper file
- `configs/hyp-rsq-loss.yaml` or equivalent hyperparameter file
- `configs/yolo26n-c1-rsq.yaml` copied from the current C1 config if the codebase expects model-specific config names
- `scripts/train_c1_rsq_loss.sh` or a markdown run recipe if scripts are not used

### Modify

- the main detection loss file, usually `ultralytics/utils/loss.py`
- the training config loader only if new hyp keys are not automatically accepted
- logging code to record RSQ component values if existing logging supports custom loss names

### Read but avoid modifying

- assignment code, usually `ultralytics/utils/tal.py`
- Detect head module
- original C1 YAML/config
- original dataset split files

## 8. Required logging

For every RSQ experiment, save or log:

- total box loss;
- MPDIoU term;
- NWD term;
- `alpha_s` mean for positives;
- SpecEdge term if enabled;
- SpecCore term if enabled;
- Quality term if enabled;
- number of positive samples;
- NaN/Inf guard status.

For evaluation, save:

- `metrics_original_protocol.csv`
- `metrics_per_class.csv`
- `latency.csv`
- `params_flops.csv`
- `small_fn_diagnostics.csv` if available
- `localization_partial_overlap.csv` if available
- `fp_reason_stats.csv` if available
- visualizations of improved and worsened cases

## 9. Required experiment matrix

Run the matrix in `scripts/run_rsq_loss_experiment_matrix.md` if present. Minimal required matrix:

| ID | Method | Spec prior | Quality | Purpose |
|---|---|---|---|---|
| L0 | C1 original loss | no | no | baseline |
| L1 | C1 + MPDIoU | no | no | geometry-only control |
| L2 | C1 + NWD | no | no | small-object smoothing control |
| L3 | C1 + MPDIoU + NWD | no | no | small-object box loss |
| L4 | C1 + RSQ-v1 | edge/core | no | specular-aware box loss |
| L5 | C1 + RSQ-v2 | edge/core | yes | full RSQ-Loss |

Optional after L5 is stable:

- `rsq_lambda_edge`: 0.02 / 0.05 / 0.10
- `rsq_lambda_core`: 0.02 / 0.05 / 0.10
- `rsq_tau`: 24 / 32 / 40
- `rsq_nwd_c`: 12.8 / 16.0 / 20.0

## 10. Safety and rejection criteria

Reject or roll back a variant if:

- AP50 drops by more than 1.0 without strong AP75/APs gain;
- APs drops;
- reflection/wave/foam false positives increase clearly;
- training becomes unstable or loss contains NaN/Inf;
- gains only appear in AP but not AP75/APs or diagnostic categories;
- runtime overhead is large due to inefficient specular mask computation.

## 11. Important implementation principles

- Keep the original C1 config unchanged. Create a new config or hyp file.
- Keep the original loss as default.
- Implement RSQ as an opt-in ablation.
- Do not change Detect, assignment, dataset split, or evaluation protocol.
- Do not introduce RAS/RGZ crop inference into this loss experiment.
- Use specular prior only if it is already available or can be exposed non-invasively.
- Start with box-only `MPDIoU + NWD`; add specular terms only after stable training is confirmed.
- Warm up specular and quality terms after several epochs.
- Compare by AP, AP50, AP75, APs, ARs, FP reason statistics, and small-FN diagnostics.

## 12. Expected conclusion format

After experiments, report results in this form:

```text
L0 C1 original:
AP=..., AP50=..., AP75=..., APs=..., ARs=..., reflection FP=..., wave/foam FP=...

L3 MPDIoU+NWD:
ΔAP=..., ΔAP75=..., ΔAPs=...
Finding: geometry-only small-object smoothing is / is not sufficient.

L4 RSQ-v1:
ΔAP=..., ΔAP75=..., ΔAPs=..., Δreflection FP=...
Finding: specular-edge/core loss is / is not useful.

L5 RSQ-v2:
ΔAP=..., ΔAP75=..., ΔAPs=..., low-confidence TP change=..., duplicate change=...
Finding: quality calibration is / is not useful.
```

Do not claim success from AP alone. The target improvements are AP75, APs, reflection false positives, and low-confidence true-positive ranking.
