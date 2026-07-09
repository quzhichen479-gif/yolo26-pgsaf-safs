# RSQ-Loss experiment matrix

This file defines the controlled experiment sequence for YOLO26-C1 + RSQ-Loss.

Do not run all variants at once. The purpose is to isolate whether the gain comes from small-object box smoothing, specular prior constraints, or score-quality calibration.

## 1. Fixed baseline protocol

Keep identical to the current C1 baseline unless the repository already has an official ablation protocol.

Must keep fixed:

- dataset split;
- image size;
- optimizer;
- base learning rate;
- epochs;
- augmentation policy;
- assignment;
- Detect head;
- evaluation protocol;
- confidence/NMS settings during validation unless the official protocol says otherwise.

Record the exact baseline command before starting.

## 2. Variants

| ID | Loss setting | SpecEdge | SpecCore | Quality | Purpose |
|---|---|---|---|---|---|
| L0 | original C1 loss | off | off | off | baseline |
| L1 | MPDIoU only | off | off | off | test CIoU replacement without NWD |
| L2 | NWD only or CIoU+NWD | off | off | off | test small-object smoothing |
| L3 | MPDIoU + NWD | off | off | off | small-object box hybrid |
| L4 | MPDIoU + NWD + SpecEdge + SpecCore | on | on | off | RSQ-v1 |
| L5 | MPDIoU + NWD + SpecEdge + SpecCore + Quality | on | on | on | RSQ-v2 |

Recommended order:

```text
L0 -> L3 smoke test -> L3 full train -> L4 -> L5
```

Only run L1 and L2 if L3 behavior is unclear.

## 3. Suggested hyperparameter files

If the repository supports separate hyp YAMLs, create:

```text
configs/hyp-c1-ciou.yaml
configs/hyp-c1-mpdiou.yaml
configs/hyp-c1-nwd.yaml
configs/hyp-c1-mpdiou-nwd.yaml
configs/hyp-c1-rsq-v1.yaml
configs/hyp-c1-rsq-v2.yaml
```

If it does not support separate hyp YAMLs, create one config and override through CLI or experiment-specific YAML copies.

## 4. Initial settings

### L3: MPDIoU + NWD

```yaml
box_loss_type: mpdiou_nwd
rsq_tau: 32.0
rsq_nwd_c: 16.0
rsq_enable_spec_edge: false
rsq_enable_spec_core: false
rsq_enable_quality: false
```

### L4: RSQ-v1

```yaml
box_loss_type: rsq
rsq_tau: 32.0
rsq_nwd_c: 16.0
rsq_lambda_edge: 0.05
rsq_lambda_core: 0.05
rsq_edge_thr: 0.25
rsq_warmup_epochs: 10
rsq_enable_spec_edge: true
rsq_enable_spec_core: true
rsq_enable_quality: false
```

### L5: RSQ-v2

```yaml
box_loss_type: rsq
rsq_tau: 32.0
rsq_nwd_c: 16.0
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
rsq_enable_quality: true
```

## 5. Smoke tests

Before full training, run:

1. one forward pass;
2. one training batch;
3. 100-iteration mini-train if the repo supports it.

Check:

```text
loss finite
no NaN
no Inf
positive sample count normal
box loss magnitude close to original range
NWD term not dominating
SpecEdge and SpecCore terms non-negative
```

Save smoke-test logs in:

```text
runs/rsq_loss_smoke/
```

## 6. Required result files

For every variant, save:

```text
metrics_original_protocol.csv
metrics_per_class.csv
latency.csv
params_flops.csv
loss_components.csv
small_fn_diagnostics.csv
localization_partial_overlap.csv
fp_reason_stats.csv
visualizations/improved_cases/
visualizations/worsened_cases/
```

If some diagnostic scripts do not exist, record `not_available` in a README inside the result directory.

## 7. Comparison table template

Fill this table after training:

| ID | AP | AP50 | AP75 | APs | ARs | reflection FP | wave/foam FP | small FN | duplicate | latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0 C1 original | | | | | | | | | | |
| L1 MPDIoU | | | | | | | | | | |
| L2 NWD | | | | | | | | | | |
| L3 MPDIoU+NWD | | | | | | | | | | |
| L4 RSQ-v1 | | | | | | | | | | |
| L5 RSQ-v2 | | | | | | | | | | |

## 8. Acceptance criteria

### Accept L3 if

```text
AP75 improves, or APs improves without AP50 collapse.
```

### Accept L4 if

```text
AP75/APs improves and reflection/wave/foam FP does not increase.
```

### Accept L5 if

```text
AP75/APs improves and low-confidence true-positive ranking or duplicate-box behavior improves.
```

## 9. Rejection criteria

Reject a variant if:

```text
AP50 drops > 1.0
APs drops
AP75 drops while AP gain is negligible
reflection/wave/foam FP increases clearly
training is unstable
runtime overhead is large due to prior-map processing
```

## 10. Final report format

Use this exact summary format:

```text
Best variant: L?
Reason:
- AP change: ...
- AP75 change: ...
- APs change: ...
- reflection FP change: ...
- wave/foam FP change: ...
- low-confidence TP / duplicate change: ...

Conclusion:
Geometry-only small-object smoothing is [useful/not useful].
Specular prior loss is [useful/not useful].
Quality calibration is [useful/not useful].
Recommended next step: [keep RSQ / tune RSQ / reject RSQ / combine with RGZ].
```
