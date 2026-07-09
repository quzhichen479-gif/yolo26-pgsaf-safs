# BUDQ-YOLO26 project design

## 1. Project position

This document defines the next loss-function direction after RSQ-Loss for the YOLO26-C1 water-surface floating waste detection task.

New mainline:

```text
YOLO26-C1 + BUDQ-YOLO26 Loss
```

Full name:

```text
Boundary-Uncertainty and Duplicate-aware Quality Loss for YOLO26
```

Chinese name:

```text
边界不确定性与重复感知质量损失
```

## 2. Why not continue RSQ-Loss

The RSQ experiment matrix produced useful signals but did not validate the full RSQ direction.

Key interpretation:

1. NWD improved high-IoU localization signals and is worth rechecking under fixed initialization.
2. MPDIoU+NWD was the most balanced same-cohort result, but it became conservative and increased low-confidence false negatives.
3. SpecEdge/SpecCore did not improve the box loss when used as explicit loss terms.
4. Quality calibration over-promoted imperfect candidates and increased poor-localization / duplicate false positives.

Therefore, the next loss should not use pseudo-specular prior as the main hammer. C1 can keep using pseudo-specular prior in feature recalibration, but the loss should be designed from detection-task failure modes:

```text
tiny-object localization instability
+ ambiguous visible boundaries
+ background spill-over
+ duplicate / poor-quality score ordering
```

## 3. YOLO26-specific constraints

BUDQ-YOLO26 must respect YOLO26's design.

Mandatory:

```text
Do not add DFL.
Do not add reg_max.
Do not add distribution bins.
Do not add a GFL-style head.
Do not modify Detect.
Do not modify assignment in Y0-Y5.
Do not change C1 structure.
```

This is important because YOLO26 has removed or avoided DFL-style distribution regression. The new loss must work on continuous box outputs.

## 4. Detection-loss reasoning

A single-stage detector loss usually has these roles:

```text
box loss: teaches geometric localization
classification loss: teaches category and foreground/background scoring
quality or ranking term: aligns score order with final prediction quality
```

For water-surface small objects, a good box loss must answer:

```text
Did the predicted box cover the visible object core?
Did it tolerate reasonable ambiguous boundary error?
Did it spill into clear background?
Did its score rank above background but below a better duplicate?
```

This is different from simply maximizing IoU against a single hard GT box.

## 5. Main formulation

BUDQ-YOLO26 uses:

```text
L_BUDQ = lambda_box * L_UBR + lambda_rank * L_DAR + L_cls
```

Where:

```text
L_UBR: Uncertainty-aware Box Regression
L_DAR: Duplicate-aware Ranking
L_cls: original YOLO26 classification loss
```

There is no DFL term.

## 6. L_UBR: uncertainty-aware box regression

### 6.1 Boundary uncertainty construction

For each ground-truth box:

```text
s = sqrt(w_gt * h_gt)
u = clamp(rho * s, u_min, u_max)
B_in  = shrink(B_gt, u)
B_out = expand(B_gt, u)
```

Interpretation:

```text
B_in: visible object core
B_gt: original annotation
B_out: tolerated boundary region
```

Recommended defaults:

```yaml
budq_rho: 0.10
budq_u_min: 1.0
budq_u_max: 4.0
```

### 6.2 Core coverage loss

```text
L_cover = 1 - area(B_pred ∩ B_in) / area(B_in)
```

This term asks whether the prediction covers the target core. It is useful for tiny floating objects, where center drift is more harmful than a slightly larger predicted box.

### 6.3 Spill loss

```text
L_spill = area(B_pred - B_out) / area(B_pred)
```

This term penalizes the predicted area that clearly leaves the tolerated target boundary. It targets boxes that expand into water texture, foam, ripple, shadow, shore clutter, or other background.

### 6.4 NWD smoothing

Use NWD as a distance smoothing term for tiny boxes:

```text
D = sqrt((cx - cx_gt)^2 + (cy - cy_gt)^2 + ((w - w_gt)^2 + (h - h_gt)^2) / 4)
L_nwd = 1 - exp(-D / C)
```

Recommended default:

```yaml
budq_nwd_c: 16.0
```

NWD should not replace all other terms. It is a smoothing signal for tiny boxes.

### 6.5 Delayed MPDIoU tightening

MPDIoU or corner-distance IoU is useful for tightening boxes, but it should not dominate early training for ambiguous tiny objects.

Use:

```text
beta_t = min(1, epoch / budq_tight_warmup_epochs)
```

Recommended default:

```yaml
budq_tight_warmup_epochs: 30
```

### 6.6 Small-object weight

```text
alpha_s = exp(-sqrt(w_gt * h_gt) / tau)
```

Recommended default:

```yaml
budq_tau: 32.0
```

### 6.7 UBR formula

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

## 7. L_DAR: duplicate-aware ranking

YOLO26 FreeNMS / weak-NMS behavior means score ordering matters. A high-score duplicate or partial-overlap box can survive and harm AP.

BUDQ therefore uses pairwise ranking, not direct quality-target calibration.

### 7.1 Positive-vs-background ranking

Select per image:

```text
positive: matched candidate with IoU >= pos_iou_thr
negative: high-score background candidate with IoU < neg_iou_thr
ignore: partial-overlap candidate in between
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
L_posneg = relu(rank_margin - score_pos + score_neg)
```

### 7.2 Duplicate ranking

For candidates assigned to the same GT:

```text
leader = highest-quality candidate
other candidates = duplicates
```

Loss:

```text
L_dup = relu(dup_margin - score_leader + score_duplicate)
```

Recommended default:

```yaml
budq_dup_margin: 0.05
```

This term should be added only after UBR and posneg ranking are stable.

## 8. Final loss

```text
L_BUDQ-YOLO26 = lambda_box * L_UBR + lambda_rank * L_DAR + L_cls
```

Recommended strategy:

```text
Y3: lambda_rank = 0
Y4: enable L_posneg only
Y5: enable L_posneg and L_dup
```

## 9. Why this is not a simple loss mix

BUDQ-YOLO26 assigns each term a specific task role:

| Term | Role |
|---|---|
| L_cover | prevent tiny-object core miss |
| L_spill | avoid background over-expansion |
| L_nwd | smooth tiny-box pixel sensitivity |
| delayed L_mpdiou | tighten boxes after early stabilization |
| L_posneg | rank true small objects above high-score background |
| L_dup | rank the best box above duplicates |

This structure directly addresses the observed L2/L3/L5 behavior:

- NWD helps AP75 but is incomplete.
- MPDIoU+NWD reduces false positives but can become conservative.
- Direct quality calibration can over-promote imperfect candidates.
- Pairwise ranking is safer than direct score target rewriting.

## 10. Suggested Y-series experiments

| ID | Method | Purpose |
|---|---|---|
| Y0 | fixed-init C1 original loss | strict baseline |
| Y1 | NWD-only / CIoU+NWD | recheck L2 under fixed init |
| Y2 | MPDIoU+NWD | recheck L3 under fixed init |
| Y3 | UBR-box | test boundary uncertainty |
| Y4 | UBR + posneg ranking | test low-confidence TP ranking |
| Y5 | UBR + posneg + duplicate ranking | test FreeNMS duplicate ordering |

## 11. Success criteria

Y3 is promising if:

```text
AP75 improves
APs does not drop
poor-localization FP decreases
FN does not clearly increase
```

Y4 is promising if:

```text
low-confidence FN decreases
high-score background FP does not increase
APs or ARs improves
precision remains stable
```

Y5 is promising if:

```text
duplicate / poor-localization FP decreases
AP75 remains stable or improves
FreeNMS output has fewer duplicate boxes
```

## 12. Paper narrative

Recommended claim if validated:

```text
To address ambiguous boundaries and score-ordering instability in YOLO26-based water-surface small-object detection, we propose BUDQ-YOLO26, a DFL-free boundary-uncertainty and duplicate-aware loss. It decomposes localization into core coverage, tolerated boundary spill, tiny-object distance smoothing, and delayed box tightening, and improves FreeNMS-oriented score ordering through pairwise duplicate-aware ranking.
```

Avoid claiming:

```text
BUDQ uses DFL.
BUDQ changes Detect.
BUDQ depends on specular prior.
BUDQ is proven effective without fixed-init multi-seed evidence.
```
