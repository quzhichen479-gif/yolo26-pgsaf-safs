# RGSA experiment matrix and acceptance checklist

This file is for Codex after adapting `modules/rgsa.py` into the real YOLO26-C1 repository.

## 0. Pre-run checks

Before training:

```text
[ ] RGSA is registered in the module registry.
[ ] C1 original YAML is unchanged.
[ ] New YAML is named yolo26n-c1-rgsa.yaml or equivalent.
[ ] Detect receives exactly [P3_RGSA, P4_C1, P5_C1].
[ ] No Detect/loss/assignment/dataloader changes.
[ ] gamma initializes to 0.
[ ] Forward-shape smoke test passes.
[ ] Params/FLOPs are logged.
```

## 1. Main controlled experiments

| ID | Model | Spec prior | Aux loss | Expected role |
|---|---|---|---|---|
| E0 | C1 baseline | C1 original | no | fixed reference |
| RGSA-A | C1 + RGSA | no | no | content-only selection ablation |
| RGSA-B | C1 + RGSA | Rcore/Eedge | no | main low-risk version |
| RGSA-C | C1 + RGSA | Rcore/Eedge | GT-small mask | supervised small-target selection |
| RGSA-D | C1 + RGSA | Rcore/Eedge | RAS selected-crop pseudo mask | select-crop distillation |

Run RGSA-A/B first. Only run RGSA-C/D if A/B do not collapse and selection maps are sparse.

## 2. Required metrics

Save the same standard metrics as C1:

```text
metrics_original_protocol.csv
metrics_per_class.csv
latency.csv
params_flops.csv
```

Also save diagnostics if available:

```text
small_fn_diagnostics.csv
localization_partial_overlap.csv
fp_reason_stats.csv
rgsa_select_stats.csv
```

## 3. RGSA selection diagnostics

Recommended `rgsa_select_stats.csv` columns:

```text
image_id
mean_select
p50_select
p75_select
p90_select
p95_select
mean_select_gt_small
mean_select_gt_medium_large
mean_select_high_rcore_empty
mean_select_eedge_high_rcore_low
select_sparsity_gt_0p5
select_sparsity_gt_0p7
```

Interpretation:

```text
mean_select_gt_small should be higher than mean_select_high_rcore_empty.
mean_select should stay sparse.
A dense select map means RGSA has degenerated into ordinary spatial attention.
```

## 4. Visualization requirements

For at least 30 validation images, save:

```text
original image
C1 predictions
RGSA predictions
S3 selection overlay
Rcore/Eedge overlay if available
```

Prioritize:

```text
improved small-object cases
worsened false-positive cases
reflection_highlight_background cases
wave/ripple cases
carton/bottle class-specific cases
```

## 5. Acceptance criteria

Accept RGSA only if all conditions hold:

```text
AP drop <= 0.2 from C1
APs >= C1 + 0.8
AP75 >= C1 + 0.4 or localization_partial_overlap decreases
reflection_highlight_background FP does not clearly increase
wave/ripple FP does not clearly increase
latency overhead < 3 ms/img
```

Strong success:

```text
AP >= C1 + 0.6
APs >= C1 + 1.5
AP75 >= C1 + 1.0
AP50 drop < 0.3
selection maps are sparse and interpretable
```

Reject if:

```text
AP50 drops > 1.0
APs drops
reflection/wave FP increases while AP gain is negligible
selection map covers most water background
gamma grows too large and destabilizes C1
```

## 6. Reporting template

Use this wording in reports:

```text
RGSA is not a replacement for RAS-v2c selective zoom because it does not increase real image-space pixels.
It is a low-latency internalization of select-crop region selection.
Its success should be judged by APs/AP75 gains, FP reason statistics, selection sparsity, and latency overhead.
```
