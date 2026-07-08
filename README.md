# YOLO26 PG-SAF / SAFS / RGSA experiments

This repository contains a Codex-ready implementation package for controlled YOLO26-C1 ablations inspired by spatial-frequency selection, specular-aware feature selection, and RAS/select-crop region selection.

## Experiments

1. **C1 + PG-SAF**
   - Goal: verify whether semantic-aligned P4-to-P3 fusion improves small-object semantic reflux.
   - Insert position: where P4 is upsampled and fused with P3.
   - Do not change Detect, loss, assignment, dataset split, or training protocol.

2. **C1 + SAFSBlock**
   - Goal: verify whether specular-aware spatial-frequency selection separates useful small-object edges from water-surface pseudo edges.
   - Insert position: after C1 P3 recalibration and before Detect.
   - Must use Rcore/Eedge gating when available.

3. **C1 + RGSA**
   - Goal: distill the region-selection part of RAS/select crop into an internal P3 selective attention module.
   - Insert position: after C1 P3 recalibration and before Detect.
   - Must remain residual-safe and must not claim to replace image-space crop/zoom.

## Files

- `CODEX_IMPLEMENTATION_PROMPT.md`: full instruction for Codex to adapt PG-SAF / SAFSBlock into the actual YOLO26-C1 repository.
- `CODEX_RGSA_IMPLEMENTATION_PROMPT.md`: full instruction for Codex to adapt RGSA selective attention into the actual YOLO26-C1 repository.
- `modules/pg_saf.py`: PG-SAF prototype.
- `modules/safs_block.py`: SAFSBlock prototype.
- `modules/rgsa.py`: RGSA prototype.
- `configs/yolo26n-c1-pgsaf.yaml`: YAML integration template.
- `configs/yolo26n-c1-safs.yaml`: YAML integration template.
- `configs/yolo26n-c1-rgsa.yaml`: YAML integration template.
- `scripts/run_experiment_matrix.md`: recommended PG-SAF / SAFS experiment matrix and acceptance criteria.
- `scripts/run_rgsa_experiment_matrix.md`: recommended RGSA experiment matrix and acceptance criteria.

## Main principle

These modules are residual-safe and C1-compatible. They are not intended to replace RAS/RGZ image-space zoom. PG-SAF and SAFS test whether semantic alignment and spatial-frequency selection can improve YOLO26-C1, while RGSA tests whether select-crop region-selection knowledge can be internalized as lightweight P3 attention without changing the detector head, loss, assignment, or evaluation protocol.
