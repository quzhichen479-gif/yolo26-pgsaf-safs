# YOLO26 PG-SAF / SAFS experiments

This repository contains a Codex-ready implementation package for two controlled YOLO26-C1 ablations inspired by SFS-DETR spatial-frequency selection ideas.

## Experiments

1. **C1 + PG-SAF**
   - Goal: verify whether semantic-aligned P4-to-P3 fusion improves small-object semantic reflux.
   - Insert position: where P4 is upsampled and fused with P3.
   - Do not change Detect, loss, assignment, dataset split, or training protocol.

2. **C1 + SAFSBlock**
   - Goal: verify whether specular-aware spatial-frequency selection separates useful small-object edges from water-surface pseudo edges.
   - Insert position: after C1 P3 recalibration and before Detect.
   - Must use Rcore/Eedge gating when available.

## Files

- `CODEX_IMPLEMENTATION_PROMPT.md`: full instruction for Codex to adapt this package into the actual YOLO26-C1 repository.
- `modules/pg_saf.py`: PG-SAF prototype.
- `modules/safs_block.py`: SAFSBlock prototype.
- `configs/yolo26n-c1-pgsaf.yaml`: YAML integration template.
- `configs/yolo26n-c1-safs.yaml`: YAML integration template.
- `scripts/run_experiment_matrix.md`: recommended experiment matrix and acceptance criteria.

## Main principle

These modules are residual-safe and C1-compatible. They are not intended to replace RAS/RGZ image-space zoom. They test whether SFS-DETR-style semantic alignment and spatial-frequency selection can improve YOLO26-C1 without changing the detector head, loss, or assignment.
