# YOLO26 PG-SAF / SAFS / RSQ-Loss experiments

This repository contains Codex-ready implementation packages for controlled YOLO26-C1 ablations and training-side improvements for water-surface floating waste detection.

## Experiments

1. **C1 + PG-SAF**
   - Goal: verify whether semantic-aligned P4-to-P3 fusion improves small-object semantic reflux.
   - Insert position: where P4 is upsampled and fused with P3.
   - Do not change Detect, loss, assignment, dataset split, or training protocol.

2. **C1 + SAFSBlock**
   - Goal: verify whether specular-aware spatial-frequency selection separates useful small-object edges from water-surface pseudo edges.
   - Insert position: after C1 P3 recalibration and before Detect.
   - Must use Rcore/Eedge gating when available.

3. **C1 + RSQ-Loss**
   - Goal: verify whether a reflectance-guided small-object quality loss improves AP75/APs and reduces reflection-induced false localization.
   - Insert position: training loss only.
   - Do not change Detect, assignment, dataset split, model structure, or evaluation protocol.
   - Main components: MPDIoU + NWD small-object box loss, specular-edge alignment, specular-core suppression, and optional quality-aware score calibration.

## Files

- `CODEX_IMPLEMENTATION_PROMPT.md`: full instruction for Codex to adapt PG-SAF / SAFSBlock into the actual YOLO26-C1 repository.
- `CODEX_RSQ_LOSS_IMPLEMENTATION_PROMPT.md`: full instruction for Codex to implement RSQ-Loss in the actual YOLO26-C1 repository.
- `docs/RSQ_LOSS_PROJECT_DESIGN.md`: RSQ-Loss method design, formula, implementation architecture, and paper narrative.
- `modules/pg_saf.py`: PG-SAF prototype.
- `modules/safs_block.py`: SAFSBlock prototype.
- `configs/yolo26n-c1-pgsaf.yaml`: YAML integration template.
- `configs/yolo26n-c1-safs.yaml`: YAML integration template.
- `scripts/run_experiment_matrix.md`: recommended PG-SAF / SAFS experiment matrix and acceptance criteria.
- `scripts/run_rsq_loss_experiment_matrix.md`: recommended RSQ-Loss experiment matrix and acceptance criteria.

## Main principle

PG-SAF and SAFSBlock are residual-safe and C1-compatible structure ablations. They are not intended to replace RAS/RGZ image-space zoom.

RSQ-Loss is a training-side loss ablation. It is not intended to change the detector head or assignment. It tests whether C1's pseudo-specular prior can improve small-object localization quality and reflection-aware score ranking during training.
