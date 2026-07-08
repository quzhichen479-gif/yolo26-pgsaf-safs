# Codex implementation prompt: YOLO26-C1 + PGW-SAFR

You are working in the existing YOLO26 water-surface floating waste detection repository. Implement a controlled C1-internal ablation named **C1 + PGW-SAFR**.

Do not rewrite the detector. Do not change Detect, loss, assignment, dataloading, evaluation protocol, or inference form. This must remain a full-image, one-pass C1 variant.

## Motivation

The validated C1 baseline already introduces a pseudo-specular prior to recalibrate YOLO26n features for water-surface mirror/specular reflection. RAS-v2c + selective zoom is useful as an upper-bound pipeline, but it is no longer a single-forward detector. PGW-SAFR is intended to improve C1 itself, not to replace or imitate RAS.

The goal is frequency-domain risk recalibration:

- use Haar DWT to split C1 feature into LL/LH/HL/HH bands;
- use C1's existing specular prior to derive or expose:
  - `Rcore`: specular-core risk, normalized [0,1];
  - `Eedge`: useful specular/target edge cue, normalized [0,1];
- suppress high-frequency response in `Rcore` regions;
- lightly preserve LH/HL edge bands in `Eedge` regions outside high-risk cores;
- keep HH conservative by default because diagonal high-frequency often contains water ripple, foam, and specular noise.

Do **not** describe this as wavelet super-resolution. It cannot restore real pixels lost by resizing/downsampling. Its role is to separate useful object-edge frequency from water-surface pseudo-frequency.

## Files from this package

Use:

- `modules/pgw_safr.py`
- `configs/yolo26n-c1-pgw-safr.yaml`
- `scripts/smoke_test_pgw_safr.py`

Copy `PGWSAFR` into the actual repository's module path, typically one of:

- `ultralytics/nn/modules/pgw_safr.py`
- or the existing custom C1 module directory.

Then register it in the same place where custom modules are exposed to YOLO's YAML parser, typically:

- `ultralytics/nn/modules/__init__.py`
- `ultralytics/nn/tasks.py`
- or the repository-specific parse_model registry.

## Required insertion point

Use the existing C1 model YAML as the base. Create a new YAML file, do not alter the original C1 YAML.

Target neck logic:

```text
P3_raw, P4_raw, P5_raw
    ↓
C1 SpecularAwareFeatureRecalibration
    ↓
P3_C1, P4_C1, P5_C1
    ↓
PGWSAFR only on P3_C1 first
    ↓
Detect([P3_PGW, P4_C1, P5_C1])
```

Only refine P3 in the first experiment. Do not add PGW-SAFR to P4/P5 until P3-only has positive evidence.

## Spec prior wiring

The module accepts:

```python
PGWSAFR(c)([feature, spec_prior])
```

where `spec_prior` is `[B, 2, H, W]`:

- channel 0 = `Rcore`
- channel 1 = `Eedge`

If the existing C1 implementation already computes a pseudo-specular map but does not expose `Rcore/Eedge`, implement the smallest non-invasive adapter:

1. Reuse the existing C1 pseudo-specular map generation.
2. Derive `Rcore` and `Eedge` using the same logic or thresholds used by C1.
3. Do not add a new expensive image preprocessing pipeline.
4. Do not change the output behavior of the original C1 module.

If YAML cannot pass `spec_prior` cleanly, create a wrapper around the existing C1 recalibration stage that internally calls PGWSAFR with the spec map already available.

## Ablation order

Run in this order:

1. **C1 baseline**
2. **C1 + PGW-Veto-P3**
   - `mode="veto"`
   - `max_edge_gain=0.0`
   - `max_core_penalty=0.25`
   - Purpose: verify whether suppressing specular-core high-frequency reduces reflection FP.
3. **C1 + PGW-SAFR-P3**
   - `mode="safr"`
   - `max_edge_gain=0.12`
   - `max_core_penalty=0.25`
   - `boost_hh=False`
   - Purpose: verify whether Eedge-guided LH/HL preservation improves AP75/APs without increasing ripple FP.
4. Optional only after positive P3 result: **P3+P4 PGW-SAFR**.

## Required smoke test

Before training:

```bash
python scripts/smoke_test_pgw_safr.py
```

Also run one YOLO model build / forward-shape test using the new YAML.

The key property to verify:

```text
gamma=0 initialization makes the new model initially behave like C1.
```

## Metrics to save

Save the same files as C1, plus diagnostics if available:

- `metrics_original_protocol.csv`
- `metrics_per_class.csv`
- `latency.csv`
- `params_flops.csv`
- `fp_reason_stats.csv`
- `small_fn_diagnostics.csv`
- `localization_partial_overlap.csv`
- visualizations of improved and worsened reflection/ripple cases

## Acceptance criteria

Minimum useful result:

- AP does not drop by more than 0.2 from C1;
- AP75 improves or localization_partial_overlap decreases;
- reflection_highlight_background FP or wave_ripple FP decreases;
- APs does not drop.

Strong result:

- AP +0.5 or more;
- AP75 +1.0 or more;
- reflection/wave FP clearly decreases;
- latency overhead is small enough to preserve one-pass deployment value.

Reject the module if:

- AP50 drops by more than 1.0;
- APs drops;
- reflection/wave FP increases while AP gain is negligible;
- duplicate or false positives clearly increase.
