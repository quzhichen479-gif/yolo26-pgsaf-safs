# Codex implementation prompt: YOLO26-C1 + BUDQ-YOLO26 Loss

You are working in the actual YOLO26 water-surface floating waste detection repository. Implement the **Y-series loss experiments** for the existing YOLO26-C1 baseline.

This prompt supersedes the earlier RSQ-Loss direction for the next loss mainline.

## 0. Core decision

The previous RSQ-Loss experiments showed that NWD / MPDIoU+NWD contain useful small-object localization signals, but SpecEdge/SpecCore and quality calibration are not beneficial as currently configured. Therefore, the next loss direction must avoid the hammer effect of forcing the C1 pseudo-specular prior into the loss.

The new direction is:

```text
BUDQ-YOLO26 Loss
Boundary-Uncertainty and Duplicate-aware Quality Loss for YOLO26
```

Chinese name:

```text
边界不确定性与重复感知质量损失
```

## 1. Absolute constraints

These constraints are mandatory.

1. **Do not add DFL.**
2. **Do not add reg_max.**
3. **Do not add distribution bins.**
4. **Do not add a GFL-style distributional box head.**
5. **Do not modify Detect.**
6. **Do not modify assignment for Y0-Y5.**
7. **Do not modify dataset split or evaluation protocol.**
8. **Keep the original YOLO26 classification loss unless the experiment explicitly enables the ranking regularizer.**
9. **Keep the original C1 architecture unchanged.**
10. **Make BUDQ opt-in. The original YOLO26-C1 loss must remain the default.**

If the actual YOLO26 codebase has removed DFL, do not reintroduce it. If a stale DFL utility exists but is unused, do not activate it.

## 2. Mandatory reading before editing code

Read these files first and summarize them in your implementation notes:

```text
README.md
CODEX_IMPLEMENTATION_PROMPT.md
CODEX_RSQ_LOSS_IMPLEMENTATION_PROMPT.md
CODEX_BUDQ_YOLO26_IMPLEMENTATION_PROMPT.md
docs/RSQ_LOSS_PROJECT_DESIGN.md
docs/BUDQ_YOLO26_PROJECT_DESIGN.md
scripts/run_rsq_loss_experiment_matrix.md
scripts/run_budq_yolo26_experiment_matrix.md
modules/budq_yolo26_loss.py
configs/hyp-c1-budq-yolo26.yaml
```

Then inspect the actual YOLO26-C1 codebase:

```bash
find . -iname "*c1*.yaml" -o -iname "*safrg*.yaml" -o -iname "*spec*.yaml"
grep -R "SpecularAwareFeatureRecalibration\|SAFRG\|Pspec\|specular\|Rcore\|Eedge" -n .
find . -path "*loss*.py" -o -path "*tal*.py" -o -path "*assigner*.py" -o -path "*train*.py"
grep -R "CIoU\|bbox_loss\|BboxLoss\|v8DetectionLoss\|TaskAlignedAssigner\|DFL\|dfl\|reg_max" -n ultralytics .
```

You must explicitly report whether YOLO26 currently has any active DFL path. If it does not, BUDQ must not add one.

## 3. Why this loss exists

The FloatingWaste-I task has these loss-level failure modes:

- tiny boxes are overly sensitive to 1-2 px offsets;
- object boundaries are uncertain because of transparent bottles, wet cartons, weak texture, partial submersion, and water contact;
- box regression can become conservative and reduce false positives while increasing low-confidence false negatives;
- FreeNMS / weak NMS makes score ordering more important, so duplicate or poor-localization boxes must not receive overly high scores;
- RSQ-v1/v2 showed that directly using specular prior in the loss can over-constrain training.

Therefore, BUDQ-YOLO26 focuses on:

```text
continuous-box boundary uncertainty
+ small-object distance smoothing
+ delayed box tightening
+ duplicate-aware ranking
```

It does **not** rely on pseudo-specular prior in the first version.

## 4. Target loss structure

The final optional loss is:

```text
L_BUDQ-YOLO26 = λ_box * L_UBR + λ_rank * L_DAR + L_cls
```

Where:

```text
L_UBR = Uncertainty-aware Box Regression
L_DAR = Duplicate-aware Ranking
L_cls = original YOLO26 classification loss
```

No DFL term is allowed.

## 5. L_UBR: uncertainty-aware box regression

Use continuous predicted boxes. Convert to `xyxy` inside the loss if necessary.

For every target box `B_gt`, compute:

```text
s = sqrt(w_gt * h_gt)
u = clamp(rho * s, u_min, u_max)
B_in  = shrink(B_gt, u)
B_out = expand(B_gt, u)
```

Recommended defaults:

```yaml
budq_rho: 0.10
budq_u_min: 1.0
budq_u_max: 4.0
```

If box coordinates are normalized, convert `u` into normalized units using the input image size. If the loss operates in pixel units, keep `u` in pixels.

### 5.1 Core coverage loss

```text
L_cover = 1 - area(B_pred ∩ B_in) / area(B_in)
```

Purpose: ensure the predicted box covers the visible object core.

### 5.2 Spill loss

```text
L_spill = area(B_pred - B_out) / area(B_pred)
```

Purpose: penalize predicted area that clearly spills beyond the tolerated boundary.

### 5.3 NWD smoothing

```text
D = sqrt((cx - cx_gt)^2 + (cy - cy_gt)^2 + ((w - w_gt)^2 + (h - h_gt)^2) / 4)
L_nwd = 1 - exp(-D / C)
```

Recommended default:

```yaml
budq_nwd_c: 16.0
```

### 5.4 Delayed MPDIoU tightening

Use MPDIoU or the closest existing corner-distance IoU helper in the repository.

```text
beta_t = min(1, epoch / budq_tight_warmup_epochs)
```

Recommended default:

```yaml
budq_tight_warmup_epochs: 30
budq_lambda_mpd: 0.5
```

### 5.5 Small-object weight

```text
alpha_s = exp(-sqrt(w_gt * h_gt) / tau)
```

Recommended default:

```yaml
budq_tau: 32.0
```

### 5.6 UBR formula

```text
L_UBR = alpha_s * (
    lambda_cover * L_cover
  + lambda_spill * L_spill
  + lambda_nwd * L_nwd
) + beta_t * lambda_mpd * L_mpdiou
```

Recommended defaults:

```yaml
budq_lambda_cover: 1.0
budq_lambda_spill: 0.5
budq_lambda_nwd: 1.0
budq_lambda_mpd: 0.5
```

## 6. L_DAR: duplicate-aware ranking

BUDQ uses ranking only after UBR is stable. Do not enable ranking in the first smoke test.

### 6.1 Positive-vs-background ranking

Inside each image, select:

```text
positive: matched positive candidates with IoU >= budq_pos_iou_thr
negative: high-score background candidates with IoU < budq_neg_iou_thr
ignore: partial-overlap candidates between the two thresholds
```

Recommended defaults:

```yaml
budq_pos_iou_thr: 0.50
budq_neg_iou_thr: 0.25
budq_rank_margin: 0.10
budq_rank_topk_neg: 32
```

Loss:

```text
L_posneg = relu(margin - score_pos + score_neg)
```

### 6.2 Duplicate ranking

For candidates assigned to the same GT, choose the highest-quality candidate as leader and rank it above duplicate candidates:

```text
L_dup = relu(margin_dup - score_leader + score_duplicate)
```

Recommended default:

```yaml
budq_dup_margin: 0.05
```

Do not enable duplicate ranking until Y4 passes.

## 7. Required implementation stages

### Y0: fixed-init C1 original loss

Create a fixed initialization protocol first. Either:

- reset all RNGs before every `YOLO(C1_yaml)` construction; or
- save one fixed `C1_init.pt` after transfer initialization and load the same init for all Y-series experiments.

Without fixed initialization, Y-series conclusions are not reliable.

### Y1: NWD-only or CIoU+NWD control

Purpose: re-check whether NWD improves AP75 under fixed initialization.

### Y2: MPDIoU+NWD control

Purpose: reproduce the useful L3 signal under fixed initialization.

### Y3: UBR-box

Purpose: test boundary uncertainty without ranking.

### Y4: UBR + posneg ranking

Purpose: reduce low-confidence true positives being outranked by high-score background.

### Y5: UBR + posneg + duplicate ranking

Purpose: adapt score ordering to YOLO26 FreeNMS / weak-NMS behavior.

## 8. Suggested files to create or modify in the actual YOLO26 repo

Create/adapt:

```text
ultralytics/utils/budq_yolo26_loss.py
configs/hyp-c1-budq-yolo26.yaml
configs/yolo26n-c1-budq.yaml if the repo uses model-specific config copies
scripts/train_y_series_budq.md or .sh
```

Modify:

```text
ultralytics/utils/loss.py or the actual YOLO26 detection loss file
training config loader only if custom hyp keys are otherwise rejected
logging to include BUDQ component values
```

Read but avoid modifying:

```text
Detect head
TaskAlignedAssigner or equivalent assignment code
C1 model YAML
C1 module implementation
```

## 9. Required logging

For every Y-series experiment, log:

```text
loss_ubr_total
loss_cover
loss_spill
loss_nwd
loss_mpdiou
alpha_s_mean
beta_t
loss_rank_posneg if enabled
loss_rank_dup if enabled
positive_count
finite_guard
```

Evaluation outputs:

```text
metrics_original_protocol.csv
metrics_per_class.csv
small_fn_diagnostics.csv
localization_partial_overlap.csv
fp_reason_stats.csv
duplicate_stats.csv
low_conf_tp_stats.csv
latency.csv
params_flops.csv
```

## 10. Acceptance criteria

Y3 / UBR is promising if:

```text
AP75 improves
APs does not drop
poor-localization FP decreases
FN does not clearly increase
```

Y4 / posneg ranking is promising if:

```text
low-confidence FN decreases
high-score background FP decreases or remains stable
APs or ARs improves
precision does not collapse
```

Y5 / duplicate ranking is promising if:

```text
duplicate / poor-localization FP decreases
AP75 does not drop
FreeNMS output has fewer duplicate boxes
```

Reject a variant if:

```text
AP50 drops by more than 1.0
APs drops clearly
FN increases sharply
poor-localization / duplicate FP increases
training becomes unstable
```

## 11. Paper-safe claim if successful

Safe claim:

```text
We propose a YOLO26-compatible boundary-uncertainty and duplicate-aware loss for water-surface small-object detection. Without reintroducing DFL or changing the detection head, it improves localization robustness by separating core coverage, boundary tolerance, and spill-over penalty, and improves FreeNMS-oriented score ordering through duplicate-aware pairwise ranking.
```

Forbidden claims:

```text
Do not claim that BUDQ uses DFL.
Do not claim that BUDQ changes Detect.
Do not claim that BUDQ improves C1 unless fixed-init multi-seed results support it.
Do not claim RSQ-v1/v2 succeeded.
Do not claim specular prior is the main reason for BUDQ.
```
