# Codex implementation prompt: YOLO26-C1 + DySample / CARAFE P4-to-P3 ablations

You are working in the actual YOLO26 water-surface floating waste detection repository. Implement controlled C1-internal structure ablations for **P4-to-P3 dynamic upsampling / fusion**.

Do not rewrite the detector. Do not change Detect, loss, assignment, dataloading, dataset split, evaluation protocol, or inference form. These experiments must remain full-image, one-pass C1 variants.

## Motivation

C1 is the YOLO26n variant with water-surface pseudo-specular prior / specular-aware recalibration. The current failure pattern suggests that ordinary attention and frequency modules may not reliably improve C1. The next lower-risk direction is to improve cross-scale semantic alignment:

```text
P4 has stronger semantics but lower spatial resolution.
P3 carries small-object localization but is vulnerable to water-surface pseudo edges.
Fixed P4->P3 upsampling may inject misaligned or contaminated context.
```

Therefore, test mature dynamic upsampling modules first:

1. **DySample-P4toP3**: learn sampling offsets instead of fixed nearest/bilinear upsample.
2. **CARAFE-P4toP3**: content-aware feature reassembly.
3. **CARAFEPlusLite-P4toP3**: stronger context/post-refinement variant, only after CARAFE is stable.

## Files from this package

Use:

- `modules/dynamic_upsample.py`
- `configs/yolo26n-c1-dysample-p4p3.yaml`
- `configs/yolo26n-c1-carafe-p4p3.yaml`
- `scripts/smoke_test_dynamic_upsample.py`

Copy `dynamic_upsample.py` into the actual repository's module path, usually one of:

- `ultralytics/nn/modules/dynamic_upsample.py`
- or the existing custom C1 module directory.

Then register exported classes in the same place where YOLO custom modules are exposed to YAML parsing, usually:

- `ultralytics/nn/modules/__init__.py`
- `ultralytics/nn/tasks.py`
- or the repository-specific `parse_model` registry.

Required classes:

- `DySample`
- `CARAFE`
- `CARAFEPlusLite`
- `C1P4P3DynamicFusion`
- aliases: `DynamicP4P3Fusion`, `CARAFEPlus`

## Preferred implementation path: direct upsample replacement

Find the C1 YAML neck stage where P4 is upsampled to P3 resolution. Replace only the upsample operator.

Before:

```text
P4_C1 -> nearest/bilinear Upsample(scale=2) -> Concat with P3_C1 -> original fusion block
```

After DySample:

```text
P4_C1 -> DySample(c=P4_channels, scale=2) -> Concat with P3_C1 -> original fusion block
```

After CARAFE:

```text
P4_C1 -> CARAFE(c=P4_channels, scale=2) -> Concat with P3_C1 -> original fusion block
```

Keep the downstream fusion block unchanged. Keep Detect inputs semantically equivalent to the original C1 model.

## Fallback path: residual dynamic P4-to-P3 injection

If direct YAML replacement is difficult because the existing YOLO parser treats upsample specially, use:

```python
C1P4P3DynamicFusion(c3, c4, upsampler="dysample")([P3_C1, P4_C1, spec_prior])
```

or:

```python
C1P4P3DynamicFusion(c3, c4, upsampler="carafe")([P3_C1, P4_C1, spec_prior])
```

Then Detect receives:

```text
Detect([P3_dynamic, P4_C1, P5_C1])
```

This wrapper has `gamma=0` initialization, so the model initially behaves like C1 when c_out == c3.

## Spec prior use

First run vanilla dynamic upsampling without spec gate:

```text
use_spec_gate = false
```

Only if the vanilla module is positive or neutral, run the spec-guided version:

```text
use_spec_gate = true
```

Spec gate should use C1's existing two-channel prior if available:

- channel 0: `Rcore`, specular-core risk;
- channel 1: `Eedge`, useful specular/target edge cue.

Do not add a new expensive image preprocessing pipeline.

## Ablation order

Run exactly in this order:

1. **C1 baseline**
2. **C1 + DySample-P4toP3**
3. **C1 + CARAFE-P4toP3**
4. **C1 + CARAFEPlusLite-P4toP3** only if CARAFE is stable
5. **C1 + SpecGate-DySample-Fusion** only if DySample is positive/neutral
6. **C1 + SpecGate-CARAFE-Fusion** only if CARAFE is positive/neutral

Do not combine DySample and CARAFE in one model.
Do not modify P5->P4 until P4->P3 has positive evidence.
Do not add P2 in this experiment.

## Smoke tests

Before training:

```bash
python scripts/smoke_test_dynamic_upsample.py
```

Then build each new YOLO YAML and run one dummy forward or one very small validation pass.

## Required outputs

For each experiment, save:

- `metrics_original_protocol.csv`
- `metrics_per_class.csv`
- `latency.csv`
- `params_flops.csv`
- `small_fn_diagnostics.csv` if available
- `localization_partial_overlap.csv` if available
- `fp_reason_stats.csv` if available, including:
  - reflection_highlight_background
  - wave_ripple
  - foam
  - shadow/reflection
  - bank_clutter
- visualizations for improved and worsened cases

## Acceptance criteria

DySample-P4toP3 is useful if:

- AP does not drop by more than 0.2;
- APs improves or ARs improves;
- AP75 improves or localization_partial_overlap decreases;
- reflection/wave FP does not increase clearly;
- latency overhead is small.

CARAFE-P4toP3 is useful if:

- AP75 or APs improves;
- reflection/wave FP does not increase;
- latency overhead is acceptable.

Reject any module if:

- AP50 drops by more than 1.0;
- APs drops clearly;
- reflection_highlight_background FP or wave_ripple FP increases while AP gain is negligible;
- latency overhead destroys the lightweight one-pass value.

## Paper positioning if positive

Do not claim that dynamic upsampling solves the real-pixel bottleneck like RAS. The correct claim is:

```text
Dynamic P4-to-P3 alignment improves how semantic context is injected into C1's small-object feature branch while preserving full-image one-pass inference.
```

If spec-gated fusion is positive, the stronger claim is:

```text
C1's mirror/specular prior can further constrain cross-scale semantic injection, reducing the risk that water-surface reflection patterns are propagated into the P3 small-object branch.
```
