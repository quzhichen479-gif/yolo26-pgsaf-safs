# YOLO26 PG-SAF / SAFS / PGW-SAFR experiments

This repository contains a Codex-ready implementation package for controlled YOLO26-C1 ablations inspired by semantic alignment, spatial-frequency selection, and pseudo-specular guided wavelet recalibration.

## Experiments

1. **C1 + PG-SAF**
   - Goal: verify whether semantic-aligned P4-to-P3 fusion improves small-object semantic reflux.
   - Insert position: where P4 is upsampled and fused with P3.
   - Do not change Detect, loss, assignment, dataset split, or training protocol.

2. **C1 + SAFSBlock**
   - Goal: verify whether specular-aware spatial-frequency selection separates useful small-object edges from water-surface pseudo edges.
   - Insert position: after C1 P3 recalibration and before Detect.
   - Must use Rcore/Eedge gating when available.

3. **C1 + PGW-SAFR**
   - Goal: verify whether C1's pseudo-specular prior can guide Haar-wavelet band recalibration.
   - Insert position: after C1 P3 recalibration and before Detect.
   - This remains a full-image, single-forward C1 variant. It is not RAS/RGZ, SAHI, or selective zoom.
   - First ablation should use `mode="veto"` to suppress high-frequency response in Rcore regions; main ablation should use `mode="safr"` to lightly preserve LH/HL edge bands using Eedge while keeping HH conservative.

## Files

- `CODEX_IMPLEMENTATION_PROMPT.md`: full instruction for Codex to adapt this package into the actual YOLO26-C1 repository.
- `modules/pg_saf.py`: PG-SAF prototype.
- `modules/safs_block.py`: SAFSBlock prototype.
- `modules/pgw_safr.py`: PGW-SAFR pseudo-specular guided Haar-wavelet prototype.
- `configs/yolo26n-c1-pgsaf.yaml`: YAML integration template.
- `configs/yolo26n-c1-safs.yaml`: YAML integration template.
- `configs/yolo26n-c1-pgw-safr.yaml`: YAML integration template.
- `scripts/run_experiment_matrix.md`: recommended experiment matrix and acceptance criteria.
- `scripts/smoke_test_pgw_safr.py`: shape/no-op smoke test for PGW-SAFR.

## Main principle

These modules are residual-safe and C1-compatible. They are not intended to replace RAS/RGZ image-space zoom. They test whether C1's full-image detector can be improved without changing the detector head, loss, assignment, or inference form.

For PGW-SAFR specifically, do not claim wavelet super-resolution. Its intended role is frequency-domain risk recalibration: suppress specular-core pseudo high-frequency and cautiously retain useful edge bands guided by the existing mirror/specular prior.
