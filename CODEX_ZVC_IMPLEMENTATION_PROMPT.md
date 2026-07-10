# Codex implementation prompt: YOLO26-C1 + training-only Zoom-View Correction

You are working in the **actual YOLO26 water-surface floating-waste detection repository**. Implement a controlled training-only method named:

```text
C1-ZVC
C1 + Zoom-View Correction
```

Chinese name:

```text
C1 + 训练期局部放大视图矫正
```

The purpose is to transfer useful information from image-space zoom crops into the full-image C1 detector **without changing the inference graph**.

---

## 0. Evidence and core decision

Known project results:

| Experiment | Train / inference | AP | AP75 | APs | ARs |
|---|---|---:|---:|---:|---:|
| E0 C1 | original full image / full image | 41.30 | 33.14 | 28.68 | 41.44 |
| E2 C1 + RAS selective | original full image / crop inference | 33.22 | 23.09 | 23.53 | 35.60 |
| E4 RAS-v2c | mixed full+crop training / full image | 37.88 | 28.93 | 26.53 | 40.31 |
| E5 RAS-v2c + RAS selective | mixed full+crop training / crop inference | 43.70 | 36.94 | 35.04 | 45.15 |

Additional facts:

- ordinary SAHI failed;
- teacher-free P3 feature super-resolution E7 failed;
- full/crop mixed training damaged full-image inference;
- real image-space crop-and-resize contains useful information;
- FloatingWaste-I is dominated by tiny/small targets, and many targets have very few pixels at 640 input;
- C1 already provides pseudo-specular recalibration before Detect.

The next mainline is therefore:

```text
full-image C1 student
+ frozen crop-view teacher used only in training
+ reliable local classification/box correction
+ optional verified reflective hard negatives
= unchanged single-pass C1 inference
```

This is a training-view correction method, not a two-pass detector.

---

## 1. Absolute constraints

These constraints are mandatory.

1. **Do not change the C1 inference model.**
2. **Do not add a crop branch to exported or validation models.**
3. **Do not run crop generation or crop detection during validation, prediction, export, benchmarking, or deployment.**
4. **Do not modify Detect.**
5. **Do not modify the existing C1 recalibration modules.**
6. **Do not modify the original assignment algorithm.**
7. **Do not replace the original YOLO26 detection loss.**
8. **Keep the full-image detection loss as the primary loss.**
9. **The teacher must be frozen, in eval mode, and executed under `torch.no_grad()`.**
10. **Do not backpropagate from the student into the teacher.**
11. **Do not globally align full-image and crop feature maps.**
12. **Do not merge full-image and crop samples as ordinary equal-status training images.**
13. **Do not add WBF, Soft-NMS, cluster fusion, or any second-stage box fusion.**
14. **ZVC must be opt-in; the original C1 training path must remain the default.**
15. **The saved inference checkpoint must contain only the student/C1 state.**
16. **Fixed initialization and fixed data split are required for comparisons.**

If the actual repository architecture makes any of these requirements ambiguous, preserve the intent: ZVC exists only as an auxiliary training signal.

---

## 2. Mandatory reading and repository inspection

Read the following package files first:

```text
README.md
CODEX_ZVC_IMPLEMENTATION_PROMPT.md
docs/ZVC_PROJECT_DESIGN.md
modules/zoom_view_correction.py
configs/hyp-c1-zvc.yaml
scripts/run_zvc_experiment_matrix.md
tests/test_zoom_view_correction.py
```

Then inspect the actual YOLO26-C1 repository:

```bash
find . -iname "*c1*.yaml" -o -iname "*safrg*.yaml" -o -iname "*spec*.yaml"
grep -R "SpecularAwareFeatureRecalibration\|SAFRG\|Pspec\|Rcore\|Eedge" -n .
find . -path "*trainer*.py" -o -path "*loss*.py" -o -path "*dataset*.py" -o -path "*augment*.py"
grep -R "class.*Trainer\|preprocess_batch\|criterion\|loss(" -n ultralytics .
grep -R "one2one\|one2many\|end2end\|FreeNMS\|NMS" -n ultralytics .
```

Before editing, write an implementation note that identifies:

- actual C1 YAML and module paths;
- actual detection trainer class;
- actual detection loss entry point;
- raw student training outputs available before post-processing;
- whether YOLO26 exposes one-to-one and one-to-many candidate tensors;
- actual image and target coordinate units;
- actual checkpoint save path and state-dict layout.

Do not guess those paths.

---

## 3. Target training graph

The required graph is:

```text
full image ----------------------> C1 student ----------------> original YOLO26 loss
    |
    +--> selected training crop --> frozen crop teacher
                                      |
                                      +--> reliable crop predictions
                                              |
                                              +--> map boxes to full coordinates
                                              +--> match to raw student candidates
                                              +--> ZVC classification correction
                                              +--> ZVC box correction

optional curated reflective empty crop
    +--> teacher confirms no confident target
    +--> weak hard-negative correction on student candidates inside that crop
```

Total loss:

```text
L_total = L_original_C1 + L_ZVC_positive + L_ZVC_hardneg
```

The initial experiment must use:

```text
L_total = L_original_C1 + L_ZVC_positive
```

Do not enable hard negatives until positive ZVC passes.

---

## 4. Teacher policy

Implement teacher selection as a configuration option.

### Teacher T1: frozen RAS-v2c crop-view teacher

This is the preferred first teacher because E5 shows that the RAS-v2c weights are useful in crop-view inference.

Requirements:

- load a user-provided checkpoint path;
- freeze all parameters;
- call `.eval()`;
- use `torch.no_grad()`;
- do not save teacher weights inside the student checkpoint unless the framework requires a temporary training wrapper;
- strip teacher state before final inference export.

### Teacher T2: frozen C1 teacher

Use this only as a controlled ablation.

### Teacher T3: EMA student teacher

Do not implement this first. It may be added only after T1/T2 are stable because EMA introduces a moving target and complicates attribution.

---

## 5. Crop policy

### 5.1 First-stage crop source: GT small-object positives only

For Z1, generate crops around annotated small objects from the current full-image batch.

Required properties:

- crop from the original image tensor before full-image resize when the data pipeline permits;
- otherwise crop from the highest-resolution tensor available and document the limitation;
- square crop;
- context scale sampled conservatively;
- bounded center and scale jitter;
- resize to the teacher input size;
- reject crops that retain less than the configured GT visible fraction;
- preserve all GT boxes that remain sufficiently visible;
- cap crops per image and per batch.

Do not create six or more crops for every image by default.

Recommended initial policy:

```yaml
zvc_crop_size: 640
zvc_max_crops_per_image: 2
zvc_context_scale_min: 2.0
zvc_context_scale_max: 3.5
zvc_center_jitter: 0.10
zvc_scale_jitter: 0.15
zvc_min_visible_fraction: 0.85
zvc_small_max_side_px: 32
```

### 5.2 Second-stage crop source: failure-driven positives

For Z2, optionally select GT objects for which the full-image student has:

- low correct-class confidence;
- poor localization;
- or no sufficiently matched candidate.

Use detached student diagnostics for selection. Do not let the selection operation carry gradients.

### 5.3 Reflective hard negatives

Enable only in Z3 or later.

A reflective hard-negative crop is eligible only if all are true:

1. it is from a curated reflective hard-negative atlas or an explicit project annotation;
2. no GT object overlaps the crop above the configured threshold;
3. the teacher has no confident detection;
4. the crop reliability score exceeds the threshold.

Brightness alone is not a negative label.

Never use unverified empty crops to suppress candidates because unlabeled floating waste may exist.

---

## 6. Reliable teacher-positive selection

Teacher predictions must be checked against crop GT before they become targets.

For every crop GT object:

1. use the GT class;
2. find the crop-teacher prediction maximizing `class_score * IoU`;
3. require:

```text
teacher class confidence >= zvc_teacher_pos_conf
teacher IoU with crop GT >= zvc_teacher_pos_iou
```

4. compute reliability:

```text
q = sqrt(confidence * IoU)
```

5. map the selected teacher box from crop coordinates to full-image coordinates.

The prototype implementation is in:

```text
modules/zoom_view_correction.py
```

Do not use unmatched teacher predictions as positive targets.

---

## 7. Student candidate matching

Use raw student candidates from the full-image training forward, before final NMS/post-processing.

For each reliable mapped teacher box:

- compute IoU against detached student candidate boxes from the same image;
- greedily select one student candidate;
- require minimum IoU;
- do not modify the original assignment result;
- use this match only for the auxiliary ZVC loss.

If YOLO26 has both one-to-one and one-to-many branches:

- start with the branch used for final inference;
- record the chosen branch;
- do not supervise both branches until the single-branch experiment is stable.

Do not match against post-NMS predictions because that would introduce non-differentiable and deployment-specific behavior into training.

---

## 8. Positive ZVC loss

Use:

```text
L_ZVC_positive =
    r(epoch) * [
        lambda_cls * L_ZVC_cls
      + lambda_box * L_ZVC_box
    ]
```

where the ramp is:

```text
r(epoch) = 0                                      before warmup
r(epoch) = min(1, progress / ramp_epochs)         after warmup
```

### 8.1 Classification correction

Use temperature-scaled sigmoid/BCE distillation because YOLO detectors generally use independent class logits:

```text
p_T = sigmoid(z_T / T)
L_ZVC_cls = BCEWithLogits(z_S / T, stopgrad(p_T)) * T^2
```

Weight each pair by teacher reliability `q`.

Do not replace the original classification target or classification loss.

### 8.2 Box correction

Start with normalized Smooth L1 between the matched full-image student box and mapped teacher box:

```text
L_ZVC_box = SmoothL1(
    B_student / [W,H,W,H],
    stopgrad(B_teacher_to_full) / [W,H,W,H]
)
```

GIoU may be an ablation, not the first default.

Do not perform dense feature imitation. Full-view and crop-view receptive fields differ, so whole-feature alignment is not justified by the current evidence.

### 8.3 Default weights

Use conservative values:

```yaml
zvc_lambda_cls: 0.25
zvc_lambda_box: 0.25
zvc_temperature: 2.0
zvc_warmup_epochs: 10
zvc_ramp_epochs: 20
```

The original detector loss must dominate early training.

---

## 9. Reflective hard-negative loss

Only after positive ZVC succeeds, add:

```text
L_ZVC_hardneg =
    r(epoch)
    * lambda_hardneg
    * reliability
    * BCEWithLogits(student_logits_inside_crop, 0)
```

Use only high-score student candidates whose centers fall inside the verified empty reflective crop.

Recommended initial weight:

```yaml
zvc_lambda_hardneg: 0.05
```

Cap the number of candidates per crop.

This is deliberately weak. Reject any implementation that converts the C1 pseudo-specular map directly into a dense negative mask.

---

## 10. Preferred implementation architecture

Adapt to the actual codebase, but prefer the following separation:

```text
ultralytics/models/yolo/detect/train_zvc.py
    ZVC-enabled trainer or training wrapper

ultralytics/utils/zoom_view_correction.py
    pure geometry, reliability, matching, and loss helpers

ultralytics/data/zvc_crop_builder.py
    online crop construction and target transforms

configs/hyp-c1-zvc.yaml
    all opt-in settings

scripts/train_c1_zvc.py
    explicit training entry
```

Do not place teacher inference inside the exported model class.

A suitable pattern is:

1. normal student full-image forward;
2. normal base loss;
3. training wrapper builds a limited crop mini-batch;
4. frozen teacher forward under no-grad;
5. helper computes ZVC losses against raw student candidates;
6. sum losses;
7. log components;
8. checkpoint only student state.

If the framework's model `loss()` method does not expose raw boxes/logits, add a training-only return path or hook with the smallest possible change. Do not change inference return values.

---

## 11. Memory and performance safeguards

Training-only crops can cause GPU memory spikes. Implement:

- max crops per image;
- max crops per batch;
- optional teacher micro-batching;
- mixed precision consistent with the main trainer;
- teacher input detached from the student graph;
- no retention of teacher feature maps;
- immediate deletion of temporary crop tensors when possible;
- logging of training-time overhead.

Recommended initial limits:

```yaml
zvc_max_crops_per_image: 2
zvc_max_crops_per_batch: 8
zvc_teacher_batch_size: 4
```

Validation and inference latency must remain identical to C1 within normal measurement noise.

---

## 12. Required finite and safety guards

Every ZVC step must handle:

- image with no GT;
- no eligible small object;
- crop with no teacher prediction;
- teacher prediction below thresholds;
- no student match;
- empty hard-negative candidate set;
- mixed precision;
- NaN/Inf teacher outputs;
- crop box touching image boundaries;
- target truncated by crop;
- normalized vs pixel coordinates.

If ZVC produces a non-finite value:

- log the incident;
- replace only the ZVC term with zero for that batch;
- do not silently alter the base loss.

---

## 13. Required logging

Per epoch or averaged per iteration:

```text
loss_base_total
loss_zvc_total
loss_zvc_pos
loss_zvc_cls
loss_zvc_box
loss_zvc_hardneg
zvc_ramp
zvc_crop_count
zvc_teacher_valid_count
zvc_positive_pair_count
zvc_hardneg_candidate_count
zvc_teacher_quality_mean
zvc_student_teacher_match_iou_mean
zvc_crop_visible_fraction_mean
zvc_finite_guard_count
zvc_teacher_forward_ms
zvc_total_train_overhead_ms
```

Evaluation outputs must include:

```text
metrics_original_protocol.csv
metrics_per_class.csv
small_fn_diagnostics.csv
reflective_fp_stats.csv
white_or_transparent_target_recall.csv
localization_partial_overlap.csv
latency.csv
params_flops.csv
```

Do not claim inference overhead from the teacher. The teacher is absent at inference.

---

## 14. Required experiment sequence

### Z0: fixed-init C1 control

- original C1;
- fixed initialization;
- original data;
- original full-image inference.

### Z1: GT-positive ZVC

- frozen teacher;
- GT small-object crops only;
- classification + box correction;
- no hard negatives.

This is the decisive first experiment.

### Z1-cls

- classification correction only.

### Z1-box

- box correction only.

These determine which signal is useful.

### Z2: failure-driven positive ZVC

- prioritize GT objects missed or weakly localized by the student;
- otherwise same as Z1.

### Z3: positive ZVC + curated reflective hard negatives

- add only verified reflective empty crops;
- very low hard-negative weight.

### Z4: high-resolution control

Run C1 and best ZVC at 896 or 960 input, subject to GPU memory.

This separates pixel-resolution benefit from transferred crop-view knowledge.

### Z5: teacher ablation

Compare frozen RAS-v2c teacher vs frozen C1 teacher.

Do not start with EMA.

---

## 15. Acceptance criteria

Z1 is promising only if:

```text
full-image AP does not clearly decline
APs or ARs improves
AP75 is stable or improves
small-object FN decreases or remains stable
reflective FP does not sharply increase
inference graph, parameters, and latency remain C1-equivalent
```

Reject or redesign if:

```text
AP50 drops by more than 1.0
APs drops
ARs drops clearly
white/transparent target recall drops
student begins suppressing bright real targets
training is unstable
teacher is accidentally present in export/inference
```

Z3 hard negatives are accepted only if reflective FP decreases without a meaningful ARs or white/transparent-target recall loss.

---

## 16. Tests required before training

Port and pass the unit tests from:

```text
tests/test_zoom_view_correction.py
```

Add integration tests for the actual codebase:

1. crop/full box mapping round trip;
2. crop GT transformation;
3. no gradient in teacher parameters;
4. student receives gradients from ZVC;
5. no ZVC loss before warmup;
6. zero loss when no reliable teacher pair exists;
7. hard negative blocked when crop contains GT;
8. C1 inference output shape unchanged;
9. exported checkpoint contains no teacher dependency;
10. original C1 path works when `zvc_enable=false`.

Run a one-batch smoke test and print all loss components.

---

## 17. Deliverables Codex must produce

1. all source changes;
2. exact changed-file list;
3. implementation note mapping package helpers to actual repository paths;
4. unit-test output;
5. one-batch smoke-test output;
6. command or IDE entry for Z0/Z1/Z1-cls/Z1-box;
7. confirmation that inference graph is unchanged;
8. confirmation that teacher parameters have no gradients;
9. confirmation that original C1 remains the default;
10. any unresolved mismatch in coordinate units or raw-output access.

Do not report an experiment as completed unless it was actually run.

---

## 18. Paper-safe claim if successful

A safe claim is:

```text
We introduce a training-only zoom-view correction strategy for a
reflectance-aware YOLO26 detector. Image-space zoom crops are used as
privileged training observations from a frozen teacher, while inference
retains the original single-pass full-image C1 architecture. Reliable
crop predictions are mapped to the full-image coordinate system to provide
local classification and localization correction without adding a crop
branch or post-processing pipeline at deployment.
```

Forbidden claims:

```text
Do not claim that ZVC increases the physical pixels available at inference.
Do not claim that ZVC reproduces all benefits of RAS selective inference.
Do not claim causality from E5 without controlled ablations.
Do not claim reflective hard-negative learning is effective before Z3 results.
Do not claim inference is two-pass.
```
