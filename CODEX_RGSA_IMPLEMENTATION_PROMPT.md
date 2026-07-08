# Codex implementation prompt: YOLO26-C1 + RGSA selective attention

You are working in the existing YOLO26 water-surface floating waste detection repository. Implement a controlled ablation called **C1 + RGSA** based on the current C1 baseline.

RGSA means **Reflectance-Guided Selective Attention**. It is derived from the idea of RAS/select crop, but it does **not** perform image-space crop or zoom during inference. Its goal is to distill the *region-selection* part of select crop into a lightweight P3 attention module.

## Non-negotiable constraints

Do not change:

- Detect head
- loss / assignment
- dataloader and dataset split
- evaluation protocol
- original C1 config file
- NMS or post-processing for the first controlled experiment

The first version must be residual-safe and C1-compatible.

## Background facts to preserve

Current validated facts from the project:

- C1 is the stable full-image baseline with specular-aware recalibration.
- Ordinary SAHI failed because dense slicing breaks water-surface context.
- RAS selective is useful because it selects local regions worth zooming.
- RAS-v2c + selective inference is the current accuracy upper bound because it increases real image-space pixels for tiny objects.
- Teacher-free feature SR and ordinary post-neck attention did not reproduce the RAS benefit.

Therefore, **do not claim RGSA replaces RAS-v2c**. RGSA only tests whether crop-style region selection can improve full-image C1 with low overhead.

## Method definition

RGSA should be implemented as:

```text
Input image
   ↓
YOLO26-C1 backbone/neck
   ↓
P3_C1, P4_C1, P5_C1
   ↓
RGSA on P3_C1 only
   ↓
Detect([P3_RGSA, P4_C1, P5_C1])
```

The central residual equation is:

```text
P3_RGSA = P3_C1 + gamma * S3 * LocalRefine(P3_C1)
```

where:

- `S3` is a sparse selection map inspired by select crop;
- `LocalRefine` is a light local/strip convolution refinement;
- `gamma` must initialize to 0;
- `S3` should be high around likely tiny objects and useful specular edges;
- `S3` should be low on isolated specular cores and large empty water regions.

## Specular prior interface

Use the existing C1 specular prior if available. RGSA expects a two-channel map:

```text
spec[:, 0] = Rcore, specular-core risk, normalized [0, 1]
spec[:, 1] = Eedge, useful specular/target edge cue, normalized [0, 1]
```

If the current C1 implementation exposes only `Pspec`, add the smallest adapter:

- derive `Rcore` from high-intensity/high-specular core response;
- derive `Eedge` from specular edge or gradient-like response;
- do not change baseline C1 behavior;
- do not introduce a new expensive preprocessing pipeline unless the repo already uses one.

If passing `spec` through YAML parsing is difficult, implement a wrapper around the existing C1 neck/module so RGSA can access the already available spec maps internally. Keep Detect inputs externally unchanged.

## Required code to adapt

Prototype file in this package:

- `modules/rgsa.py`

Create or adapt inside the actual YOLO26 repository:

- `ultralytics/nn/modules/rgsa.py` or the repository's equivalent module path
- module registry / `__init__.py`
- model parser registration if needed
- config file copied from C1, named like `yolo26n-c1-rgsa.yaml`

Do not overwrite the original C1 YAML.

## Implementation stages

### Stage 0: smoke test

1. Add RGSA module and register it.
2. Copy C1 YAML to `yolo26n-c1-rgsa.yaml`.
3. Insert RGSA after C1 P3 recalibration and before Detect.
4. Run a forward-shape test with dummy input.
5. Confirm Detect receives exactly three tensors with the same shapes as C1 except P3 contents.
6. Confirm `gamma=0` makes initial output near C1 identity.

### Stage 1: pure residual RGSA

Train with the same settings as C1.

Use:

```text
P3_RGSA = P3_C1 + gamma * S3 * LocalRefine(P3_C1)
```

No auxiliary loss yet. This isolates the architectural effect.

### Stage 2: GT-small selection supervision

Only if Stage 1 is stable, add a weak auxiliary supervision on `S3`.

Positive selection regions:

- small/tiny GT boxes expanded by 1.3–1.8 times;
- low-confidence true positives if existing diagnostics provide them.

Negative selection regions:

- empty high-Rcore regions;
- known reflection/wave false-positive regions if diagnostic labels exist.

Suggested loss:

```text
L_total = L_det
        + lambda_select * BCE(S3, M_select)
        + lambda_sparse * mean(S3)
        + lambda_core * mean(S3 * Rcore)
```

Start with:

```text
lambda_select = 0.03 or 0.05
lambda_sparse = 0.003 or 0.005
lambda_core = 0.01 or 0.02
```

If adding training-loop loss is risky, skip Stage 2 and finish Stage 1 only.

### Stage 3: RAS-selection pseudo mask supervision

Only if historical selected crop logs exist.

Build pseudo masks from selected crop windows:

- positive center area of RAS-selected positive crops;
- downweight huge windows;
- negative mask from selected hard-negative windows with no GT, especially Rcore-high areas.

This is the closest version to “select-crop distillation”.

## YAML integration template

Use `configs/yolo26n-c1-rgsa.yaml` in this package only as a conceptual template. Codex must adapt indices to the real C1 YAML.

Expected insert position:

```text
... 
P3_C1 = C1 specular-aware recalibrated P3
P4_C1 = C1 specular-aware recalibrated P4
P5_C1 = C1 specular-aware recalibrated P5
P3_RGSA = RGSA(P3_C1, optional spec)
Detect([P3_RGSA, P4_C1, P5_C1])
```

## Experiment matrix

Run the following controlled experiments:

| ID | Model | Auxiliary selection loss | Purpose |
|---|---|---|---|
| E0 | C1 baseline | no | fixed reference |
| RGSA-A | C1 + RGSA, no spec | no | content-only selection ablation |
| RGSA-B | C1 + RGSA, Rcore/Eedge | no | main residual architecture |
| RGSA-C | C1 + RGSA, Rcore/Eedge | GT-small mask | test supervised selection |
| RGSA-D | C1 + RGSA, Rcore/Eedge | RAS pseudo mask | select-crop distillation |

Do not combine with PG-SAF or SAFSBlock until RGSA alone is evaluated.

## Required outputs

For each run, save:

- trained weights
- metrics_original_protocol.csv
- metrics_per_class.csv
- latency.csv
- params_flops.csv
- small_fn_diagnostics.csv if available
- localization_partial_overlap.csv if available
- fp_reason_stats.csv if available
- rgsa_select_stats.csv
- visualization folder:
  - improved small-object cases
  - worsened reflection/wave false-positive cases
  - selection map overlays if possible

## RGSA-specific diagnostics

Add a lightweight diagnostic script or logging function if possible:

```text
mean(S3)
percentile(S3): p50/p75/p90/p95
mean(S3 on GT-small boxes)
mean(S3 on high-Rcore empty regions)
mean(S3 on Eedge-high Rcore-low regions)
```

A successful selection map should satisfy:

```text
S3(GT-small) > S3(background)
S3(Eedge high, Rcore low) > S3(Rcore high, no GT)
mean(S3) remains sparse
```

## Acceptance criteria

RGSA is accepted only if:

- AP does not drop more than 0.2 from C1;
- APs improves by at least +0.8;
- AP75 improves by at least +0.4, or localization_partial_overlap decreases;
- reflection_highlight_background FP does not increase clearly;
- wave/ripple FP does not increase clearly;
- latency overhead is less than 3 ms/img.

Strong success:

- AP +0.6 or more;
- APs +1.5 or more;
- AP75 +1.0 or more;
- no AP50 collapse;
- selection maps are visually sparse and aligned with small objects or useful edges.

Reject RGSA if:

- AP50 drops by more than 1.0;
- APs drops;
- reflection/wave FP increases while AP gain is negligible;
- selection map becomes dense over the entire water background;
- gamma grows aggressively and destabilizes C1 behavior.

## Reporting rule

When reporting results, state clearly:

```text
RGSA does not increase real image-space pixels.
It only injects select-crop region-selection knowledge into full-image C1.
Therefore its target is low-latency APs/AP75 improvement, not replacing RAS-v2c selective zoom.
```
