# YOLO26 PG-SAF / SAFS / RSQ-Loss / BUDQ-YOLO26 / C1-ZVC experiments

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
   - Current status: not recommended as the next mainline in its full SpecEdge/SpecCore/quality form.

4. **C1 + BUDQ-YOLO26 Loss**
   - Goal: test a YOLO26-compatible, DFL-free loss for water-surface tiny-object boundary uncertainty and FreeNMS-oriented duplicate-aware score ordering.
   - Insert position: training loss only.
   - Do not add DFL, reg_max, distribution bins, or a distributional box head.
   - Do not change Detect, assignment, dataset split, model structure, or evaluation protocol.
   - Main components: uncertainty-aware box regression with core coverage, spill penalty, NWD smoothing, delayed MPDIoU tightening, and optional duplicate-aware pairwise ranking.

5. **C1 + Zoom-View Correction (C1-ZVC)**
   - Goal: transfer reliable image-space zoom-view knowledge into the full-image C1 student while keeping inference identical to C1.
   - Insert position: training wrapper / auxiliary loss only.
   - A frozen crop teacher is used only during training; validation, prediction, export, and deployment remain single-pass full-image C1.
   - Do not modify Detect, assignment, C1 recalibration, the base loss, or inference post-processing.
   - First experiment uses GT small-object positive crops only; reflective hard negatives are a later controlled extension.

## Files

- `CODEX_IMPLEMENTATION_PROMPT.md`: full instruction for Codex to adapt PG-SAF / SAFSBlock into the actual YOLO26-C1 repository.
- `CODEX_RSQ_LOSS_IMPLEMENTATION_PROMPT.md`: full instruction for Codex to implement RSQ-Loss in the actual YOLO26-C1 repository.
- `CODEX_BUDQ_YOLO26_IMPLEMENTATION_PROMPT.md`: full instruction for Codex to implement the Y-series BUDQ-YOLO26 loss experiments.
- `CODEX_ZVC_IMPLEMENTATION_PROMPT.md`: full instruction for Codex to implement training-only C1-ZVC in the actual YOLO26-C1 repository.
- `docs/RSQ_LOSS_PROJECT_DESIGN.md`: RSQ-Loss method design, formula, implementation architecture, and paper narrative.
- `docs/BUDQ_YOLO26_PROJECT_DESIGN.md`: BUDQ-YOLO26 method design, YOLO26 constraints, formulas, and paper narrative.
- `docs/ZVC_PROJECT_DESIGN.md`: C1-ZVC evidence boundary, objective, crop curriculum, formulas, and inference-invariance requirements.
- `modules/pg_saf.py`: PG-SAF prototype.
- `modules/safs_block.py`: SAFSBlock prototype.
- `modules/budq_yolo26_loss.py`: DFL-free BUDQ-YOLO26 PyTorch loss prototype.
- `modules/zoom_view_correction.py`: tested ZVC geometry, teacher filtering, matching, positive distillation, and hard-negative helpers.
- `configs/yolo26n-c1-pgsaf.yaml`: YAML integration template.
- `configs/yolo26n-c1-safs.yaml`: YAML integration template.
- `configs/hyp-c1-budq-yolo26.yaml`: BUDQ-YOLO26 hyperparameter template.
- `configs/hyp-c1-zvc.yaml`: C1-ZVC hyperparameter and safety template.
- `scripts/run_experiment_matrix.md`: recommended PG-SAF / SAFS experiment matrix and acceptance criteria.
- `scripts/run_rsq_loss_experiment_matrix.md`: recommended RSQ-Loss experiment matrix and acceptance criteria.
- `scripts/run_budq_yolo26_experiment_matrix.md`: recommended Y-series BUDQ-YOLO26 experiment matrix and acceptance criteria.
- `scripts/run_zvc_experiment_matrix.md`: controlled Z0-Z5 C1-ZVC experiment sequence and diagnostics.
- `tests/test_zoom_view_correction.py`: unit tests for coordinate mapping, crop geometry, reliable-teacher rejection, student gradients, and hard-negative guards.

## Main principle

PG-SAF and SAFSBlock are residual-safe and C1-compatible structure ablations. They are not intended to replace RAS/RGZ image-space zoom.

RSQ-Loss is a training-side loss ablation. It is not intended to change the detector head or assignment. Current RSQ evidence suggests that NWD / MPDIoU+NWD are useful signals, while the full SpecEdge/SpecCore/quality form should not continue as the next mainline without redesign.

BUDQ-YOLO26 is explicitly YOLO26-compatible and DFL-free. It works on continuous box outputs, preserves Detect and assignment, and first validates boundary uncertainty before adding ranking.

C1-ZVC treats image-space zoom as a privileged training observation, not a deployment branch. Its first priority is to preserve C1 full-image performance while transferring reliable local classification and localization signals from a frozen crop-view teacher. It does not claim to recreate physical pixels that are absent at full-image inference.
