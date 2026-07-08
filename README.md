# YOLO26 C1 improvement packages

This repository contains Codex-ready implementation packages for controlled YOLO26-C1 experiments on FloatingWaste-I water-surface floating-waste detection.

The original package covered PG-SAF / SAFSBlock feature-side ablations. The new package adds **C1-CVRC**, a crop-view rectification route derived from C1 + RAS-v2c evidence.

## Current implementation packages

### 1. C1 + PG-SAF

- Goal: verify whether semantic-aligned P4-to-P3 fusion improves small-object semantic reflux.
- Insert position: where P4 is upsampled and fused with P3.
- Do not change Detect, loss, assignment, dataset split, or training protocol.

### 2. C1 + SAFSBlock

- Goal: verify whether specular-aware spatial-frequency selection separates useful small-object edges from water-surface pseudo edges.
- Insert position: after C1 P3 recalibration and before Detect.
- Must use Rcore/Eedge gating when available.

### 3. C1-CVRC / optional C1-RGZ

- Goal: transfer RAS-style crop-view localization ability back into C1 without making crop inference mandatory.
- Training path: full-image C1 + selected original-image crop resize + crop-view rectification loss.
- Default inference: full-image single-pass C1-CVRC.
- Optional inference: C1-RGZ with reflectance-guided zoom crops and spec-aware fusion.
- Main target metrics: AP75, APs, localization partial overlap, and small-object box quality.

## Files

### Existing PG-SAF / SAFSBlock

- `CODEX_IMPLEMENTATION_PROMPT.md`: full instruction for Codex to adapt PG-SAF / SAFSBlock into the actual YOLO26-C1 repository.
- `modules/pg_saf.py`: PG-SAF prototype.
- `modules/safs_block.py`: SAFSBlock prototype.
- `configs/yolo26n-c1-pgsaf.yaml`: YAML integration template.
- `configs/yolo26n-c1-safs.yaml`: YAML integration template.
- `scripts/run_experiment_matrix.md`: recommended PG-SAF / SAFSBlock experiment matrix.

### New C1-CVRC package

- `CODEX_C1_CVRC_IMPLEMENTATION_PROMPT.md`: complete Codex task prompt for C1-CVRC / optional C1-RGZ.
- `docs/C1_CVRC_DESIGN.md`: design rationale, architecture, losses, metrics, and stop conditions.
- `modules/cvrc_selector.py`: crop selector, GT/prediction crop generation, spec-prior scoring, coordinate mapping helpers.
- `modules/crop_view_rectifier.py`: rectification loss, specular hard-negative loss, optional Zoom-Aware Spatial Gate.
- `configs/yolo26n-c1-cvrc.yaml`: C1-CVRC config template.
- `configs/yolo26n-c1-rgz-infer.yaml`: optional two-pass RGZ inference template.
- `scripts/run_c1_cvrc_experiment_matrix.md`: E9-A/E9-B/E9-C/E9-D/E9-E run matrix and acceptance criteria.

## Main principle

C1-CVRC should not become ordinary SAHI, feature-space super-resolution, or another free attention block.

The intended mechanism is:

```text
C1 full-image baseline
  -> select a few useful original-image crop views during training
  -> resize crop views so tiny objects get real pixel evidence
  -> map crop-view predictions back
  -> rectify full-view small-object localization
  -> keep default inference single-pass unless RGZ is explicitly enabled
```

Do not change Detect, loss assignment, dataset split, or the original C1 config. Create C1 copies/configs and keep all new behavior switchable.
