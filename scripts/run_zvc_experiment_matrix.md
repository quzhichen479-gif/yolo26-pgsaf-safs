# C1-ZVC controlled experiment matrix

## 1. Fixed protocol

All experiments must use:

- FloatingWaste-I original split;
- the same evaluation script and thresholds;
- the same C1 YAML;
- the same transfer initialization or saved fixed initialization;
- the same seed list;
- the same epoch, optimizer, augmentation, and image-size settings unless the experiment explicitly changes image size;
- full-image validation and inference only.

Never evaluate ZVC using crop inference.

## 2. Pre-flight checks

Before Z0/Z1:

```text
[ ] locate actual C1 YAML and best weights
[ ] locate RAS-v2c teacher checkpoint
[ ] verify teacher crop performance can be reproduced
[ ] verify box coordinate units
[ ] verify raw inference-branch student candidates are accessible in training
[ ] pass ZVC unit tests
[ ] pass one-batch finite smoke test
[ ] prove teacher parameters have grad=None
[ ] prove eval/predict/export do not call crop builder or teacher
```

## 3. Experiment table

| ID | Description | Teacher | Positive crop | cls | box | hardneg | Input |
|---|---|---|---|---:|---:|---:|---:|
| Z0 | fixed-init C1 | none | none | 0 | 0 | 0 | 640 |
| Z1-cls | classification-only ZVC | RAS-v2c | GT small | 1 | 0 | 0 | 640 |
| Z1-box | box-only ZVC | RAS-v2c | GT small | 0 | 1 | 0 | 640 |
| Z1 | core positive ZVC | RAS-v2c | GT small | 1 | 1 | 0 | 640 |
| Z2 | failure-driven ZVC | RAS-v2c | weak/missed GT | 1 | 1 | 0 | 640 |
| Z3 | positive + reflective hard negatives | RAS-v2c | Z2 + curated negatives | 1 | 1 | 1 | 640 |
| Z4-C1 | high-resolution C1 control | none | none | 0 | 0 | 0 | 896/960 |
| Z4-ZVC | high-resolution best ZVC | best | best | best | best | best | 896/960 |
| Z5-C1T | teacher ablation | frozen C1 | GT small | 1 | 1 | 0 | 640 |

Do not start Z2/Z3 until Z1 results are available.

## 4. Suggested initial seed policy

Use at least three seeds for Z0 and Z1 before declaring the direction effective.

For every seed, start from the same transfer-initialized checkpoint or reconstruct initialization under fully reset RNG state.

Report:

```text
mean
standard deviation
best
worst
paired seed difference vs Z0
```

## 5. Required logs

Training:

```text
loss_base_total
loss_zvc_total
loss_zvc_cls
loss_zvc_box
loss_zvc_hardneg
zvc_crop_count
zvc_teacher_valid_count
zvc_positive_pair_count
zvc_teacher_quality_mean
zvc_match_iou_mean
zvc_train_overhead_ms
```

Evaluation:

```text
AP
AP50
AP75
APs
ARs
per-class AP/AP50
small FN
reflective FP/image
white/transparent target recall
latency
params
FLOPs
```

## 6. Decision rules

### Continue from Z1 when

```text
AP is stable or higher
APs or ARs improves
AP75 is stable or higher
small FN does not worsen
inference latency and parameters equal C1
```

### Choose cls or box branch when

```text
Z1-cls > Z1-box: prioritize score correction and refine matching
Z1-box > Z1-cls: prioritize localization correction and crop geometry
both weak individually but Z1 works: keep joint signal and test interaction
all fail: stop ZVC before adding hard negatives
```

### Continue to Z3 only when

```text
positive ZVC passes
curated hard-negative labels are audited
teacher empty-crop confirmation is reliable
```

### Reject Z3 when

```text
reflective FP decreases but ARs drops
white/transparent real-target recall drops
AP50 drops > 1.0
```

## 7. Required diagnostic comparisons

### Pixel-information diagnostic

Compare:

```text
C1-640
ZVC-640
C1-896/960
ZVC-896/960
```

Interpretation:

- high-resolution C1 gain much larger than ZVC-640 gain: physical sampling remains dominant;
- ZVC-640 gain without high-resolution cost: crop knowledge is partially transferable;
- ZVC and high resolution add: both sampling and decision correction matter.

### Teacher diagnostic

Compare frozen RAS-v2c and frozen C1 teachers under identical crop policy.

Do not infer teacher superiority from their full-image metrics; evaluate teacher reliability on the actual crop set.

## 8. Output directory convention

Recommended:

```text
runs/zvc/Z0_c1_seedXX/
runs/zvc/Z1_cls_seedXX/
runs/zvc/Z1_box_seedXX/
runs/zvc/Z1_joint_seedXX/
runs/zvc/Z2_failure_seedXX/
runs/zvc/Z3_hardneg_seedXX/
runs/zvc/Z4_highres_seedXX/
runs/zvc/Z5_teacher_c1_seedXX/
```

Each directory must contain:

```text
args.yaml
weights/best.pt
weights/last.pt
results.csv
zvc_train_stats.csv
metrics_original_protocol.csv
metrics_per_class.csv
small_fn_diagnostics.csv
reflective_fp_stats.csv
latency.csv
```

## 9. Final report template

| Model | AP | AP50 | AP75 | APs | ARs | small FN | reflective FP/img | latency | params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Z0 C1 | | | | | | | | | |
| Z1-cls | | | | | | | | | |
| Z1-box | | | | | | | | | |
| Z1 joint | | | | | | | | | |
| Z2 | | | | | | | | | |
| Z3 | | | | | | | | | |
