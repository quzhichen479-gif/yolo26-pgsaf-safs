# YOLO26 C1 training-side experiment packages

This repository contains Codex-ready implementation packages for controlled YOLO26-C1 ablations and training-side improvements for water-surface floating-waste detection.

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
   - Current status: the full SpecEdge/SpecCore/quality form is not recommended as the mainline.

4. **C1 + BUDQ-YOLO26 Loss**
   - Goal: test a YOLO26-compatible, DFL-free loss for tiny-object boundary uncertainty and FreeNMS-oriented score ordering.
   - Do not add DFL, reg_max, distribution bins, or change Detect/assignment.

5. **C1 + Zoom-View Correction (C1-ZVC)**
   - Original goal: transfer image-space zoom-view predictions into the full-image C1 student while keeping inference unchanged.
   - Current status: **Z1 rejected**. AP/AP50/AP75/APs decreased; the box term collapsed numerically and the auxiliary gradient entered a detached one-to-one route.
   - Keep the files as an experiment record and geometry reference. Do not continue direct teacher-box/full-class-vector distillation as the mainline.

6. **C1-GACQ: GT-Anchored Cross-View Quality Learning**
   - Current recommended controlled direction.
   - Uses non-detached C1-recalibrated P3/P4/P5 features through a training-only auxiliary head.
   - GT remains the only box/class target.
   - The frozen crop teacher estimates supervision reliability only.
   - Correct-class and auxiliary quality scores are aligned to detached IoU, with intra-GT pairwise ranking.
   - The teacher, crop builder, hooks and auxiliary head are removed for validation/export/deployment, leaving the original single-pass C1 graph.

## Main C1-GACQ entry files

- `CODEX_GACQ_IMPLEMENTATION_PROMPT.md`: complete Codex execution instruction for the actual YOLO26-C1 repository.
- `docs/GACQ_PROJECT_DESIGN.md`: evidence boundary, architecture, formulas, gradient requirements, risks and paper-safe claim.
- `modules/gacq_training_route.py`: tested detector-agnostic PyTorch prototype for the auxiliary head, fixed GT points, reliability, localization, quality and ranking losses.
- `configs/hyp-c1-gacq.yaml`: configuration and safety template.
- `scripts/run_gacq_experiment_matrix.md`: G0-G6 causal ablation order, acceptance criteria and stop conditions.
- `tests/test_gacq_training_route.py`: tensor, reliability, non-detached feature-gradient, ramp and checkpoint-strip tests.

## Other files

- `CODEX_IMPLEMENTATION_PROMPT.md`: PG-SAF / SAFSBlock integration instruction.
- `CODEX_RSQ_LOSS_IMPLEMENTATION_PROMPT.md`: RSQ-Loss implementation instruction.
- `CODEX_BUDQ_YOLO26_IMPLEMENTATION_PROMPT.md`: BUDQ-YOLO26 implementation instruction.
- `CODEX_ZVC_IMPLEMENTATION_PROMPT.md`: original C1-ZVC implementation record.
- `docs/RSQ_LOSS_PROJECT_DESIGN.md`
- `docs/BUDQ_YOLO26_PROJECT_DESIGN.md`
- `docs/ZVC_PROJECT_DESIGN.md`
- `modules/pg_saf.py`
- `modules/safs_block.py`
- `modules/budq_yolo26_loss.py`
- `modules/zoom_view_correction.py`
- `configs/yolo26n-c1-pgsaf.yaml`
- `configs/yolo26n-c1-safs.yaml`
- `configs/hyp-c1-budq-yolo26.yaml`
- `configs/hyp-c1-zvc.yaml`
- `scripts/run_experiment_matrix.md`
- `scripts/run_rsq_loss_experiment_matrix.md`
- `scripts/run_budq_yolo26_experiment_matrix.md`
- `scripts/run_zvc_experiment_matrix.md`
- `tests/test_zoom_view_correction.py`

## C1-GACQ implementation principle

```text
full-image C1 shared features before detach
+ training-only auxiliary head
+ fixed GT-anchored localization
+ crop-teacher reliability weighting
+ IoU-consistent class/quality scoring and ranking
= original C1 inference after stripping the auxiliary route
```

C1-GACQ is an unvalidated research hypothesis until the gradient audit, disabled-equivalence tests and paired G0-G4 experiments pass. Do not claim performance improvement from the design or prototype alone.
