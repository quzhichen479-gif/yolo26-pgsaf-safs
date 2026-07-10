# C1-GACQ Controlled Experiment Matrix

## 0. Purpose

This matrix isolates whether the benefit comes from:

1. GT-anchored localization on non-detached shared features;
2. localization-quality scoring;
3. intra-GT candidate ranking;
4. crop-teacher reliability;
5. scale-consistent crop curriculum.

Do not run the final combined method first.

## 1. Pre-run hard gates

### Gate A: repository audit

Document:

- exact C1 YAML and checkpoint;
- exact P3/P4/P5 recalibrated modules entering Detect;
- where YOLO26 creates `x.detach()` for one-to-one;
- active box format and coordinate units;
- whether DFL is active (do not add it if absent);
- training/validation/export construction paths.

### Gate B: disabled equivalence

With `gacq_enable=false`:

- forward outputs equal C1 under fixed inputs;
- base losses equal C1;
- state dict key set equals C1;
- validation metrics match within deterministic tolerance;
- params/FLOPs/export graph equal C1.

### Gate C: gradient audit

Backpropagate only GACQ loss for one batch. Require finite non-zero gradients in backbone, neck, C1 and auxiliary head. Teacher must have no gradient.

### Gate D: loss-scale audit

For at least 100 batches:

- localization and quality losses are finite and O(1)-scale before weighting;
- localization/quality gradient norm ratio is between 0.25 and 4.0;
- auxiliary/base total-loss ratio remains below 0.10 initially;
- no component is six orders of magnitude below another due only to coordinate units.

### Gate E: teacher/crop scale audit

Compare teacher training slices and GACQ crops:

- resized GT width/height distributions;
- context ratio;
- truncation rate;
- class distribution;
- teacher correct-class confidence and IoU-to-GT.

Do not tune performance before this audit is saved.

## 2. Paired initialization

Use the same fixed C1 initialization and seed set for all experiments. Minimum final evidence:

```text
seeds = [42, 43, 44]
```

A single seed may be used only for smoke rejection.

## 3. Experiments

### G0: paired C1 control

- Current codebase, same seed and evaluation protocol.
- No GACQ components instantiated.
- Purpose: exclude implementation-era drift.

### G1: non-detached GT localization only

Enable:

```text
training-only auxiliary head
fixed GT points
GIoU + NWD localization
```

Disable:

```text
crop teacher
quality branch loss
ranking
```

Purpose: prove that the auxiliary route and localization supervision can improve or preserve C1 without teacher information.

Reject G1 if AP50 drops >1.0, APs drops, carton AP50 collapses, or shared-feature gradients are absent.

### G2: G1 + localization-quality scoring

Enable correct-class IoU target and auxiliary quality logit. Still no crop teacher and no ranking.

Purpose: test the Oracle-indicated score/IoU consistency opportunity.

Required diagnostics:

- score-IoU Spearman correlation;
- ECE/Brier for correct-class quality;
- low-quality high-score candidate count;
- AP75 and APs.

### G3: G2 + intra-GT pairwise ranking

Use center and axial neighbor candidates for each GT. Rank candidates according to detached IoU to GT.

Purpose: test whether better-localized candidates rise above nearby poorer candidates without changing NMS/FreeNMS.

Do not add background ranking in G3.

### G4: G3 + crop-teacher reliability

Enable the frozen crop teacher. Teacher outputs only the detached reliability weight.

Ablations within G4:

```text
G4a: confidence + IoU-to-GT + visibility
G4b: G4a + augmentation stability
G4c: G4b + student evidence gate
```

Proceed sequentially. G4 is useful only if it improves over G3 under paired seeds.

### G5: G4 + scale-consistent crop curriculum

Match crop-resized object size and context distributions to the verified teacher training distribution. Start with GT-small crops, then optionally select failure-driven crops where full-view evidence exists but localization/quality is poor.

Do not add reflective hard negatives in this matrix.

### G6: optional training-only sparse cross-scale fusion

Only after G4/G5 passes. Within the auxiliary head, align a local P4 feature to the GT-centered P3 location and learn a dynamic scale weight. Keep it training-only.

This is a secondary experiment, not the first implementation target.

## 4. Main endpoints

Primary:

```text
AP50
APs
```

Secondary:

```text
AP
AP75
ARs
carton AP50
bottle AP50
```

Mechanism endpoints:

```text
candidate recall@0.3/@0.5
mean best IoU
score-IoU correlation
IoU-oracle gap
low-quality high-score count
duplicate count
shared-feature gradient norms
```

Deployment endpoints:

```text
student params
student FLOPs
latency
export graph/state_dict equality
```

## 5. Acceptance criteria

A stage is promising only if, across paired seeds:

- AP50 does not decrease by more than 1.0 point;
- APs does not decrease;
- ARs does not clearly decrease;
- carton degradation is not hidden by aggregate metrics;
- mechanism metric moves in the intended direction;
- deployment params/FLOPs remain exactly C1 after stripping.

Preferred success:

```text
AP/AP75/APs improve
score-IoU correlation improves
low-quality high-score candidates decrease
no inference cost increase
```

## 6. Required outputs

```text
metrics_original_protocol.csv
metrics_per_class.csv
candidate_recall_oracle.csv
quality_calibration.csv
score_iou_correlation.csv
gradient_audit.csv
teacher_reliability_by_class_size.csv
crop_scale_audit.csv
loss_component_history.csv
latency.csv
params_flops.csv
checkpoint_key_diff.txt
export_graph_diff.txt
```

## 7. Stop conditions

Stop and fix implementation before more training if:

- GACQ gradients do not reach shared C1 features;
- teacher receives gradients;
- localization loss collapses numerically;
- auxiliary/base loss ratio exceeds 0.10 without explicit approval;
- disabled mode differs from C1;
- stripped model cannot load into the original C1 class with `strict=True`;
- validation/export invokes crop generation or auxiliary head.
