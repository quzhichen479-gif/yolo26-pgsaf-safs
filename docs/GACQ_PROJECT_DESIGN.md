# C1-GACQ Project Design

## 1. Name

**C1-GACQ: GT-Anchored Cross-View Quality Learning**  
中文：**C1 + GT 锚定跨视图质量学习**

GACQ is a training-only auxiliary learning method for the existing YOLO26-C1 detector. It uses crop-view observations as privileged training information while preserving exactly the original full-image C1 inference graph.

## 2. Evidence boundary

The design is based on the current project evidence, not on an assumption that crop distillation is already successful.

- E0/C1 full-image remains the valid single-pass baseline.
- RAS-v2c + selective crop inference showed that image-space zoom can contain useful information.
- RAS-v2c full-image evaluation showed that naive full/crop mixed training damages full-image performance.
- Z1 C1-ZVC was rejected: AP, AP50, AP75 and APs decreased.
- Z1 box loss collapsed numerically relative to classification loss.
- Z1 auxiliary supervision entered the YOLO26 one-to-one branch built from `x.detach()`, so it could not update backbone, neck or C1 recalibration.
- Oracle candidate recall is already high, while GT refinement and IoU-aware re-ranking show large upper bounds. The main opportunity is localization quality and score ordering, not another generic neck fusion block.

GACQ is therefore a new controlled hypothesis. It must not be described as validated before paired multi-seed experiments.

## 3. Core architecture

```text
Full image
    |
YOLO26 backbone + neck + C1 recalibration
    |
    +---------------- original Detect route ----------------> base detection loss
    |
    +-- non-detached P3/P4/P5 shared features
              |
       training-only GACQ auxiliary head
              |
       GT-anchored fixed candidate points
              |
       localization + quality + ranking losses

Crop image -> frozen teacher -> reliability only
GT box/class ----------------> true localization/class anchors
```

At validation/export/deployment:

```text
Full image -> original C1 -> original Detect -> original output
```

The crop teacher, crop builder, feature hooks and auxiliary head are absent.

## 4. Six mandatory principles

### 4.1 Shared features must not be detached

The auxiliary route must read C1-recalibrated P3/P4/P5 tensors before the YOLO26 one-to-one `x.detach()` operation. A loss-only backward test must show finite non-zero gradients in:

- backbone;
- neck;
- C1 specular recalibration modules;
- GACQ auxiliary head.

If these gradients are absent, no full training is allowed.

### 4.2 GT is the only box and class target

The teacher must not replace ground truth.

```text
box target   = GT box
class target = GT class
```

Crop teacher boxes and full class vectors are not distilled. This prevents teacher localization bias and the annotation/teacher conflict observed in direct prediction distillation.

### 4.3 Teacher provides reliability, not labels

For each GT, map it into the crop coordinate system and compute:

- correct-class teacher confidence `c_t`;
- teacher box IoU to GT `u_t`;
- visible fraction `v_t`;
- optional augmentation stability `s_t`;
- optional full-view student evidence `m_s`.

After thresholding, use a detached geometric mean:

```text
r_t = geometric_mean(c_t, u_t, v_t, s_t, m_s)
```

`r_t` weights the auxiliary losses. It does not lower the desired quality target. A reliable teacher says “this crop contains trustworthy privileged evidence”; it does not redefine the GT.

### 4.4 Distillation points are fixed by GT geometry

For each GT, choose one FPN level from its pixel scale and sample a small fixed set:

```text
center + left + right + up + down
```

The first implementation uses at most five points per GT. Do not match free teacher/student predictions to choose the distillation location. This removes candidate drift and makes the supervision interface reproducible.

### 4.5 Localization uses pixel-scale stable losses

Z1 normalized SmoothL1 collapsed. GACQ decodes continuous `l/t/r/b` distances into pixel-coordinate boxes and uses:

```text
L_loc = GIoU(pred, GT) + lambda_nwd * NWD(pred, GT)
```

No DFL, `reg_max`, distribution bins or Detect modification is allowed.

### 4.6 Correct-class score must reflect localization quality

For each auxiliary candidate:

```text
q_i = stop_gradient(IoU(box_i, GT))
```

Only the GT class logit is supervised toward `q_i`. An auxiliary quality logit is also supervised toward `q_i`.

```text
score_i = sqrt(sigmoid(cls_i,y) * sigmoid(quality_i))
```

Candidates assigned to the same GT are pairwise ranked by `q_i`. Better-localized boxes must score higher than worse-localized neighbors.

Teacher reliability weights these terms; it is not multiplied into `q_i`.

## 5. Loss

For candidate `i` of GT `g`:

```text
L_loc_i = r_g * [GIoU_i + lambda_nwd * NWD_i]
L_q_i   = r_g * [BCE(cls_y_i, q_i) + BCE(quality_i, q_i)]
```

For two candidates of the same GT where `q_i - q_j` exceeds a threshold:

```text
L_rank_ij = (q_i - q_j) * relu(margin - score_i + score_j)
```

Total:

```text
L_total = L_base_C1
        + ramp(epoch) * [lambda_loc * L_loc
                         + lambda_quality * L_q
                         + lambda_rank * L_rank]
```

The base YOLO26-C1 loss remains unchanged.

## 6. Why this is not a conventional feature-fusion module

GACQ does not globally concatenate P2/P3/P4 or replace the neck. It combines four kinds of training information at fixed target locations:

1. full-view shared C1 features;
2. exact GT geometry and class;
3. crop-view teacher reliability;
4. localization-derived quality ordering.

The innovation target is the supervision and routing interface, not an additional deployment feature block.

## 7. Integration strategy

Preferred implementation in the actual repository:

1. Inspect the C1 YAML and find the three recalibrated tensors entering Detect.
2. Register training-only forward hooks on these three modules, or add an explicit training-only feature return that is impossible to activate in validation/export.
3. Verify captured tensors have `requires_grad=True` and are not detached copies.
4. Instantiate `GACQAuxiliaryHead` in the trainer/wrapper, not in the C1 inference YAML.
5. Run the normal full-image C1 forward and base loss.
6. Build fixed GT points from the captured feature shapes.
7. Run the frozen crop teacher under `eval()` and `torch.no_grad()` only to compute reliability.
8. Compute GACQ losses and add them after warmup/ramp.
9. Strip all GACQ and teacher keys from deployment checkpoints.
10. Validate and export using a freshly loaded original C1 model with the stripped student weights.

Do not attach the auxiliary loss to the existing detached one-to-one feature branch.

## 8. Required gradient audit

Before any 200-epoch run, execute a one-batch audit with only `L_GACQ` enabled.

Record parameter-group gradient norms:

```text
backbone_grad_norm
neck_grad_norm
c1_grad_norm
aux_head_grad_norm
original_detect_grad_norm
```

Hard requirements:

- backbone/neck/C1/aux gradients are finite and non-zero;
- teacher gradients are exactly absent;
- localization-to-quality gradient norm ratio is within a pre-registered reasonable band, initially 0.25–4.0;
- no normalized-box `1e-7`-scale collapse.

## 9. Required logging

Training logs:

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
backbone_grad_norm
neck_grad_norm
c1_grad_norm
aux_grad_norm
finite_guard_count
teacher_time_ms
aux_time_ms
```

Evaluation must include AP, AP50, AP75, APs, ARs, per-class metrics, candidate recall, quality calibration, duplicate statistics, latency, params and FLOPs.

## 10. Risks and guards

### Privileged pixels cannot be recreated

Crop views may reveal pixels absent in the full view. GACQ must supervise only samples with usable full-view evidence when that gate is enabled. Claims must be limited to representation/quality learning, not pixel reconstruction.

### Teacher reliability can become a selection bias

Report valid/rejected GT distributions by class and size. The teacher gate must not silently reject carton or transparent/white objects disproportionately.

### Quality learning can suppress recall

The quality target is IoU, while reliability is only a loss weight. Do not set the target to `IoU * reliability`; that would force low scores for difficult samples.

### Auxiliary head may learn but fail to transfer

Ablate G1 GT-only, G2 quality, G3 ranking and G4 teacher reliability. Teacher value is established only if G4 improves over G3 under paired seeds.

## 11. Paper-safe statement if successful

A safe claim after paired multi-seed validation:

> We introduce a training-only GT-anchored cross-view quality learning route for YOLO26-C1. Privileged crop observations estimate supervision reliability, while ground truth anchors localization and class identity. A non-detached auxiliary head transfers the signal into shared C1 features and aligns correct-class scores with localization quality. The auxiliary route is removed after training, leaving the original single-pass C1 inference graph unchanged.

Do not claim success before the controlled matrix passes.
