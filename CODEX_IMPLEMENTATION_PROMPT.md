# Codex implementation prompt: YOLO26-C1 + PG-SAF / SAFSBlock

You are working in the existing YOLO26 water-surface floating waste detection repository. Implement two controlled ablations based on the existing C1 baseline. Do not rewrite the detector, do not change Detect, loss, assignment, dataloading, or evaluation protocol.

## Background

Current validated facts:

- C1 is the stable full-image baseline with specular-aware feature recalibration.
- Ordinary SAHI failed on this task.
- RAS-v2c + selective zoom is the current accuracy upper bound because it increases real image-space pixels for tiny objects.
- Teacher-free feature SR and ordinary post-neck attention have failed or underperformed.

Therefore these two experiments must be conservative, residual-safe, and C1-compatible.

## Experiment 1: C1 + PG-SAF

Goal: verify whether semantic-aligned fusion helps P4-to-P3 small-object semantic reflux.

Insert position:

- At the YOLO26 neck location where P4 is upsampled and fused with P3.
- Replace or wrap the original P3/P4 fusion with PGSAF.
- Do not change later Detect inputs except that the P3 feature becomes PG-SAF refined P3.

Module behavior:

- Input: P3 high-resolution feature, P4 semantic feature, optional spec prior.
- Align P4 to P3 spatial size.
- Perform channel-wise semantic alignment.
- Perform spatial alignment gate.
- Use Rcore/Eedge prior to suppress specular-core semantic injection and preserve useful edge regions.
- Residual scale gamma must initialize to 0.

Required code:

- Add PGSAF module. The prototype is in `modules/pg_saf.py`.
- Register it wherever YOLO26 parses custom modules.
- Add a config named like `yolo26n-c1-pgsaf.yaml` based on the current C1 config.

Constraints:

- Do not change Detect.
- Do not change loss.
- Do not change assignment.
- Do not change dataset split.
- Keep training hyperparameters identical to C1 unless a repository standard ablation protocol already exists.

Evaluation focus:

- AP
- AP50
- AP75
- APs
- small FN
- localization_partial_overlap
- latency

## Experiment 2: C1 + SAFSBlock

Goal: verify whether specular-aware spatial-frequency selection can distinguish small-object edges from water-surface pseudo edges.

Insert position:

- After C1's P3 specular-aware recalibration and before Detect.
- Only refine P3. Keep P4_C1 and P5_C1 unchanged.

Module behavior:

- Input: P3_C1 and optional spec prior.
- Spatial branch: local DWConv + horizontal strip DWConv + vertical strip DWConv.
- Frequency branch: FFT/rFFT based feature mixing.
- Use Rcore/Eedge gate:
  - Eedge high: allow useful edge enhancement.
  - Rcore high: suppress specular-core enhancement.
- Residual scale gamma must initialize to 0.

Required code:

- Add SAFSBlock module. The prototype is in `modules/safs_block.py`.
- Register it wherever YOLO26 parses custom modules.
- Add a config named like `yolo26n-c1-safs.yaml` based on the current C1 config.

Constraints:

- Do not change Detect.
- Do not change loss.
- Do not change assignment.
- Do not change dataset split.
- Keep training hyperparameters identical to C1.

Evaluation focus:

- AP
- AP50
- AP75
- APs
- reflection_highlight_background FP
- wave/ripple FP
- latency

## Spec prior wiring

Use the existing C1 specular prior if available. The module expects a two-channel map:

- channel 0: Rcore, specular core risk, normalized [0,1]
- channel 1: Eedge, useful specular/target edge cue, normalized [0,1]

If the current C1 implementation does not expose Rcore/Eedge, implement the smallest non-invasive adapter:

- reuse C1's Pspec / specular map generation code;
- derive or expose Rcore and Eedge without changing the C1 baseline behavior;
- do not use a new expensive image preprocessing pipeline unless already present.

If passing spec maps through YAML parse_model is difficult, implement a wrapper around the existing C1 module/neck that internally calls PGSAF or SAFSBlock with the already available spec maps. Keep the external Detect interface identical.

## Required files to create or modify

Create:

- `ultralytics/nn/modules/pg_saf.py` or the repository's equivalent module path.
- `ultralytics/nn/modules/safs_block.py` or equivalent.
- `configs/yolo26n-c1-pgsaf.yaml`
- `configs/yolo26n-c1-safs.yaml`
- optional combined config only after single-module experiments run successfully.

Modify:

- module registry / `__init__.py`
- model parser if custom module parsing requires explicit registration
- C1 YAML copy only; do not alter original C1 config

## Required run output

For each experiment, save:

- trained weights
- metrics_original_protocol.csv
- metrics_per_class.csv
- latency.csv
- params_flops.csv
- small_fn_diagnostics.csv if available
- localization_partial_overlap.csv if available
- fp_reason_stats.csv if available
- visualization folder for improved and worsened cases

## Acceptance criteria

PG-SAF is accepted only if:

- AP does not drop more than 0.2 from C1;
- APs improves by at least +0.5;
- AP75 improves or localization_partial_overlap decreases;
- reflection/wave FP does not increase obviously.

SAFSBlock is accepted only if:

- APs or AP75 improves;
- reflection_highlight_background FP or wave/ripple FP decreases;
- no AP50 collapse;
- latency overhead is justified.

Reject the module if:

- AP50 drops by more than 1.0;
- APs drops;
- reflection/wave FP increases while AP gain is negligible;
- duplicate/false positives increase clearly.

## Important implementation principles

- Start with gamma=0 residual initialization, confirming the new model initially behaves like C1.
- First run a forward-shape smoke test before training.
- Compare parameter count and FLOPs.
- Train PG-SAF and SAFSBlock separately before combining them.
- Do not claim success from AP only. Check APs, AP75, FP reason statistics, and small-FN diagnostics.
