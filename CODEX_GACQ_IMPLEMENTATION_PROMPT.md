# Codex implementation prompt: YOLO26-C1 + C1-GACQ

You are working in the **actual YOLO26 water-surface floating-waste detection repository**. Implement a new controlled training-only method:

```text
C1-GACQ
GT-Anchored Cross-View Quality Learning
C1 + GT 锚定跨视图质量学习
```

The method must use non-detached shared C1 features, exact GT localization anchors, crop-teacher reliability weights and localization-quality-consistent score ranking. The complete auxiliary route must be absent during validation, prediction, export and deployment.

---

## 0. Core decision

The previous C1-ZVC Z1 run was rejected. Do not continue its direct teacher prediction distillation design.

Known mechanism failures:

1. normalized SmoothL1 box loss collapsed to a negligible scale relative to classification;
2. full teacher class-vector BCE dominated the auxiliary objective;
3. the auxiliary loss used the YOLO26 one-to-one route created from `x.detach()`, so gradients did not reach backbone, neck or C1 recalibration;
4. crop-teacher boxes were used as targets even though exact GT boxes were already available;
5. classification score learning was not aligned to localization quality.

The new design is:

```text
full-image C1 shared features before detach
+ training-only auxiliary head
+ fixed GT-anchored points
+ GT box/class targets
+ frozen crop teacher used only as a reliability estimator
+ correct-class score aligned with IoU
+ intra-GT quality ranking
= original single-pass C1 inference after training
```

---

## 1. Absolute constraints

These requirements are mandatory.

1. Do not change the C1 inference YAML or deployment graph.
2. Do not attach the auxiliary route to the existing `x.detach()` one-to-one tensors.
3. Do not modify Detect.
4. Do not modify the original YOLO26 assignment.
5. Do not replace the original C1 base detection loss.
6. Do not add DFL, `reg_max`, distribution bins or a distributional box head.
7. Do not use teacher boxes as localization targets.
8. Do not use teacher full class vectors as classification targets.
9. Do not run crop generation or teacher inference in validation/predict/export/benchmark/deployment.
10. Do not save the crop teacher or auxiliary head in the deployment checkpoint.
11. With `gacq_enable=false`, behavior must be equivalent to the original C1.
12. The stripped student checkpoint must load into the original C1 class with `strict=True`.

If any constraint cannot be met in the current codebase, stop implementation and report the exact blocking file/function instead of silently changing the method.

---

## 2. Mandatory reading

Read and summarize these files before editing the actual detector repository:

```text
README.md
CODEX_ZVC_IMPLEMENTATION_PROMPT.md
docs/ZVC_PROJECT_DESIGN.md
modules/zoom_view_correction.py
configs/hyp-c1-zvc.yaml
scripts/run_zvc_experiment_matrix.md
CODEX_GACQ_IMPLEMENTATION_PROMPT.md
docs/GACQ_PROJECT_DESIGN.md
modules/gacq_training_route.py
configs/hyp-c1-gacq.yaml
scripts/run_gacq_experiment_matrix.md
tests/test_gacq_training_route.py
```

Then inspect the actual YOLO26-C1 codebase. At minimum locate:

```bash
find . -iname "*c1*.yaml" -o -iname "*safrg*.yaml" -o -iname "*spec*.yaml"
grep -R "SpecularAwareFeatureRecalibration\|SAFRG\|Pspec\|Rcore\|Eedge" -n .
grep -R "x.detach()\|one2one\|one2many\|end2end" -n ultralytics .
find . -path "*head*.py" -o -path "*loss*.py" -o -path "*trainer*.py" -o -path "*model*.py"
grep -R "DFL\|dfl\|reg_max\|TaskAlignedAssigner\|assigner" -n ultralytics .
```

In the implementation report, state:

- exact C1 model YAML;
- exact C1 recalibration module and layer indices/names;
- exact tensors entering Detect and their strides/channels;
- exact location of one-to-one detach;
- active box representation and coordinate units;
- whether active DFL exists;
- exact trainer/loss integration points;
- exact validation/export construction paths.

Do not guess any of these.

---

## 3. Required architecture

### 3.1 Shared feature tap

Capture the three C1-recalibrated features entering Detect:

```text
P3_C1 / stride 8
P4_C1 / stride 16
P5_C1 / stride 32
```

They must be captured **before** the one-to-one `x.detach()` operation.

Preferred option:

- register forward hooks during GACQ trainer initialization on the three verified C1 recalibration modules;
- clear the feature cache before every batch;
- capture only when `model.training and gacq_enable`;
- remove hooks during validation/export and on trainer teardown.

Alternative option:

- add an explicit training-only return from the internal forward path, but only if it cannot affect the normal model API and export graph.

Do not modify the C1 YAML to add the auxiliary head.

After the student forward, assert:

```python
len(shared_features) == 3
all(f.requires_grad for f in shared_features)
all(f.grad_fn is not None for f in shared_features)
```

### 3.2 Training-only auxiliary head

Adapt `modules/gacq_training_route.py::GACQAuxiliaryHead` to the actual feature channels.

For each level output:

```text
continuous l/t/r/b distances: 4 channels
class logits: nc channels
quality logit: 1 channel
```

The head belongs to the trainer/wrapper, not the deployment C1 model. Its optimizer parameters must be registered during training.

### 3.3 Fixed GT candidate points

For every GT in full-image pixel coordinates:

1. compute `sqrt(width * height)`;
2. select P3/P4/P5 using configured thresholds;
3. select center plus axial neighboring cells;
4. use at most five candidates per GT in the first implementation.

Do not choose distillation points through teacher/student prediction matching.

### 3.4 GT-anchored continuous localization

Decode positive l/t/r/b distances around each fixed point in **pixel coordinates**.

Localization targets are always mapped full-image GT boxes:

```text
L_loc = GIoU(pred_box, GT_box) + lambda_nwd * NWD(pred_box, GT_box)
```

Do not use normalized SmoothL1 as the only localization term.

### 3.5 Crop-teacher reliability

Reuse the verified ZVC crop geometry utilities where correct, but do not reuse direct prediction distillation.

Teacher requirements:

```python
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)
with torch.no_grad():
    teacher_output = teacher(crops)
```

For each crop/GT calculate:

```text
teacher correct-class confidence
teacher best box IoU to crop-mapped GT
GT visible fraction
optional two-view teacher stability
optional full-view student evidence
```

Apply thresholds, then use a detached geometric mean reliability.

The reliability is a **loss weight only**.

Forbidden:

```text
box_target = teacher_box
class_target = teacher_class_vector
quality_target = IoU * teacher_reliability
```

Required:

```text
box_target = GT
class_identity = GT class
quality_target = stop_gradient(IoU(pred, GT))
loss_weight = teacher_reliability
```

### 3.6 Localization-quality-consistent scoring

At each auxiliary candidate, supervise only the correct GT class logit and the auxiliary quality logit toward detached IoU:

```text
q_i = stop_gradient(IoU(pred_i, GT))
L_quality = BCE(cls_logit_i[y], q_i) + BCE(quality_logit_i, q_i)
```

Use the combined auxiliary score:

```text
score_i = sqrt(sigmoid(cls_y_i) * sigmoid(quality_i))
```

For candidates assigned to the same GT, if `q_i - q_j` exceeds the configured quality gap, require:

```text
score_i >= score_j + margin
```

Do not add background ranking in the first mainline.

### 3.7 Total loss

```text
L = L_original_C1
  + ramp(epoch) * [lambda_loc * L_loc
                   + lambda_nwd * L_nwd
                   + lambda_quality * L_quality
                   + lambda_rank * L_rank]
```

Keep the original C1 loss values and logging intact.

---

## 4. Implementation stages

Implement switches so the following experiments can be run without code changes.

### G0: paired C1 control

```text
gacq_enable = false
```

No GACQ object should be required for inference or validation.

### G1: GT localization-only auxiliary route

```text
teacher = false
quality = false
ranking = false
localization = true
```

For G1 use reliability weight `1.0` for all eligible GT.

### G2: G1 + quality scoring

```text
teacher = false
quality = true
ranking = false
```

### G3: G2 + intra-GT ranking

```text
teacher = false
quality = true
ranking = true
```

### G4: G3 + crop-teacher reliability

```text
teacher = true
quality = true
ranking = true
```

Teacher usefulness is established only by G4 versus G3 under paired seeds.

### G5: G4 + scale-consistent crop curriculum

Enable only after a saved scale audit.

Do not implement reflective hard-negative training in this task.

---

## 5. Suggested files in the actual YOLO26 repository

Adapt names to the real project layout after inspection.

Create:

```text
ultralytics/utils/gacq_training_route.py
ultralytics/models/yolo/detect/gacq_trainer.py   # or actual trainer location
configs/hyp-c1-gacq.yaml
tests/test_gacq_gradient_route.py
tests/test_gacq_inference_equivalence.py
tests/test_gacq_checkpoint_strip.py
tools/audit_gacq_gradients.py
tools/audit_gacq_crop_scale.py
scripts/train_gacq_matrix.py or documented IDE entry script
```

Modify minimally:

```text
trainer initialization: register auxiliary head/optimizer params/hooks
training step: capture shared features and add optional auxiliary loss
checkpoint save: strip training-only keys from deployment weights
logging: add GACQ components
```

Avoid modifying:

```text
C1 model YAML
Detect head implementation
assignment code
base detection loss semantics
validation predictor
exporter
post-processing
```

---

## 6. Required tests before training

### Test 1: disabled equivalence

With fixed seed/input and `gacq_enable=false`:

- output tensors match original C1;
- base losses match;
- model parameter count and state-dict keys match;
- validation/export do not instantiate teacher or auxiliary head.

### Test 2: non-detached gradient route

Backpropagate only GACQ loss. Assert finite non-zero gradients in:

```text
backbone
neck
C1 recalibration
GACQ auxiliary head
```

Assert no teacher gradient.

This test is the most important test in the task.

### Test 3: GT anchoring

Perturb teacher boxes while holding teacher reliability constant. GACQ localization targets and loss target boxes must remain unchanged.

### Test 4: reliability semantics

Reducing teacher reliability must reduce loss weight but must not reduce the IoU quality target itself.

### Test 5: quality ranking

Construct two candidates for the same GT where candidate A has higher IoU than B but lower score. Ranking loss must be positive and backpropagate in the correct direction.

### Test 6: checkpoint stripping

After stripping, loading into the original C1 model with `strict=True` must succeed. No keys containing these concepts may remain:

```text
gacq_aux
crop_teacher
gacq_teacher
feature_hook
```

### Test 7: inference isolation

During validation/predict/export, add counters/assertions proving:

```text
crop_builder_calls == 0
teacher_forward_calls == 0
aux_head_forward_calls == 0
```

### Test 8: finite and scale checks

Run 100 batches and assert:

- no NaN/Inf;
- localization loss is not `1e-7` relative to quality loss due to coordinate normalization;
- auxiliary/base total loss ratio stays under the configured initial maximum;
- gradient norm ratio is logged.

Run the prototype tests as well:

```bash
pytest -q tests/test_gacq_training_route.py
```

---

## 7. Required diagnostics

Log per epoch and save CSV:

```text
gacq_loss
gacq_loc
gacq_nwd
gacq_quality
gacq_rank
gacq_iou_mean
gacq_target_quality_mean
gacq_reliability_mean
gacq_valid_gt_count
gacq_candidate_count
gacq_rank_pair_count
gacq_ramp
teacher_conf_mean
teacher_iou_gt_mean
visible_fraction_mean
teacher_reject_rate_by_class
teacher_reject_rate_by_size
backbone_grad_norm
neck_grad_norm
c1_grad_norm
aux_grad_norm
teacher_time_ms
aux_time_ms
finite_guard_count
```

Quality diagnostics:

```text
score-IoU Spearman correlation
ECE/Brier score
low-quality high-score candidate count
candidate recall@0.3/@0.5
mean best IoU
IoU-oracle gap
duplicate count
```

Always report bottle and carton separately.

---

## 8. Training protocol

1. Re-run G0 in the current codebase with seed 42.
2. Run G1/G2/G3 smoke experiments with seed 42.
3. Reject immediately if AP50 drops more than 1.0, APs drops, carton AP50 collapses or mechanism metrics worsen.
4. Only after G3 passes, load the crop teacher and run G4.
5. Run final candidates on paired seeds `[42, 43, 44]`.
6. Use the original evaluation protocol exactly:

```text
imgsz=640
batch=8
conf=0.001
IoU=0.7
max_det=300
same split and evaluation scripts
```

Do not select a different metric protocol to make GACQ look better.

---

## 9. Completion report

At completion provide:

1. files created/modified;
2. exact feature-capture layers and why they are pre-detach;
3. gradient audit table;
4. disabled equivalence result;
5. checkpoint-strip and strict-load result;
6. unit-test output;
7. smoke-training output if executed;
8. remaining risks/TODOs;
9. exact IDE or Python entry commands for G0–G4.

Do not claim an AP improvement unless actual paired evaluation supports it.

---

## 10. Implementation reference

Use these files as the specification and tensor-level starting point:

```text
modules/gacq_training_route.py
configs/hyp-c1-gacq.yaml
docs/GACQ_PROJECT_DESIGN.md
scripts/run_gacq_experiment_matrix.md
tests/test_gacq_training_route.py
```

The prototype has detector-agnostic tensor helpers. It is not permission to guess the actual YOLO26 integration points.
