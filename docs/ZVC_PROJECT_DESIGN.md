# C1-ZVC project design

## 1. Method position

C1-ZVC is a **training-only privileged-view correction method** for the existing YOLO26-C1 full-image detector.

It does not replace C1 and does not package RAS as an inference pipeline. Instead, image-space zoom crops are used only during training to teach the full-image student how to improve selected tiny-object predictions.

```text
Training:
full image -> C1 student -> base loss
zoom crop  -> frozen teacher -> reliable local correction

Inference:
full image -> original C1 -> detections
```

## 2. Motivation grounded in existing results

The observed project results show:

- C1 full-image inference remains the strongest single-pass baseline;
- direct RAS crop inference with the original C1 weights is worse than C1 full-image inference;
- mixed full/crop training damages full-image evaluation;
- matched crop-view training and crop-view inference can improve APs and AP75;
- feature-space super-resolution did not reproduce the image-space crop benefit.

The supported engineering conclusion is not that crop inference is universally necessary. It is that crop views contain useful local supervision, while naive distribution mixing and feature replacement are unsafe.

C1-ZVC therefore separates:

- the deployment distribution: full image only;
- the privileged training observation: selected zoom crops;
- the base objective: unchanged C1 detection loss;
- the correction objective: reliability-filtered local distillation.

## 3. Variables

For one training image:

- `I`: full image;
- `S(I)`: C1 student raw predictions;
- `C_k(I)`: selected image-space crop `k`;
- `T(C_k)`: frozen teacher predictions on the resized crop;
- `M_k`: crop-to-full coordinate transform;
- `q_j`: reliability of teacher pair `j`;
- `r(e)`: epoch ramp weight.

The original C1 loss is `L_base`.

## 4. Reliable positive construction

A teacher prediction may supervise the student only when it agrees with an annotated crop object.

For crop GT `g`, choose:

```text
t* = argmax_t p_t(y_g) * IoU(b_t, b_g)
```

Accept only if:

```text
p_t*(y_g) >= tau_c
IoU(b_t*, b_g) >= tau_i
```

Teacher reliability is:

```text
q = sqrt(p_t*(y_g) * IoU(b_t*, b_g))
```

The teacher box is mapped back to full-image coordinates:

```text
b_T_full = M_k(b_T_crop)
```

The matching operation is detached and does not replace the original detector assignment.

## 5. Positive correction

### 5.1 Classification correction

With temperature `T`:

```text
p_T = sigmoid(z_T / T)
L_zvc_cls = weighted_BCEWithLogits(z_S / T, stopgrad(p_T)) * T^2
```

Each pair is weighted by teacher reliability `q`.

### 5.2 Box correction

The first implementation uses normalized Smooth L1:

```text
L_zvc_box = weighted_SmoothL1(
    b_S / [W,H,W,H],
    stopgrad(b_T_full) / [W,H,W,H]
)
```

This avoids scale dependence across 640/896/960 experiments.

### 5.3 Warmup and ramp

```text
r(e) = 0                                           if e < warmup
r(e) = min(1, (e - warmup + 1) / ramp_epochs)     otherwise
```

```text
L_ZVC_positive = r(e) * (
    lambda_cls * L_zvc_cls
  + lambda_box * L_zvc_box
)
```

The full first-stage loss is:

```text
L = L_base + L_ZVC_positive
```

## 6. Reflective hard-negative extension

This extension is intentionally delayed.

A crop is eligible only when:

- it contains no GT;
- it comes from a curated reflective hard-negative source;
- the teacher has no confident target;
- the crop reliability is high.

For selected high-score student candidates inside the crop:

```text
L_ZVC_hardneg =
    r(e) * lambda_hardneg * reliability
    * BCEWithLogits(student_logits, 0)
```

The final optional objective is:

```text
L = L_base + L_ZVC_positive + L_ZVC_hardneg
```

No dense `bright pixel = background` supervision is allowed.

## 7. Why feature alignment is excluded

Crop and full views differ in:

- sampling density;
- receptive field;
- context;
- object scale;
- background proportion.

Therefore, forcing whole P3/P4/P5 feature maps to match can transfer view-specific artifacts and repeat the failure mode of feature-space enhancement. C1-ZVC aligns only reliable semantic logits and boxes for selected objects.

## 8. Why normal mixed training is excluded

Treating crop images as ordinary training samples changes the empirical training distribution. Existing E4 results show that the resulting weight set can lose full-image performance.

C1-ZVC keeps every optimization step anchored by the normal full-image loss. Crop observations are auxiliary and reliability weighted.

## 9. Crop curriculum

### Stage A: GT-positive crops

This is the cleanest causal test.

### Stage B: failure-driven positive crops

Prefer small objects with low confidence or poor localization in the detached student output.

### Stage C: curated reflective hard negatives

Add only after Stage A/B show stable gains.

## 10. Expected benefit and hard limit

ZVC may transfer:

- better class confidence for weak small targets;
- better relative box placement;
- better local discrimination learned from zoom views.

ZVC cannot create physical image samples that do not exist in the full-image input. A high-resolution control is necessary to determine how much of E5 comes from pixel sampling rather than transferable knowledge.

## 11. Main ablations

| ID | Positive cls | Positive box | Hard negative | Crop selection | Purpose |
|---|---:|---:|---:|---|---|
| Z0 | no | no | no | none | fixed-init C1 |
| Z1-cls | yes | no | no | GT small | classification signal |
| Z1-box | no | yes | no | GT small | localization signal |
| Z1 | yes | yes | no | GT small | core ZVC |
| Z2 | yes | yes | no | failure-driven | targeted correction |
| Z3 | yes | yes | yes | positive + curated reflection | FP control |
| Z4 | best | best | best | best | high-resolution control |
| Z5 | yes | yes | no | GT small | teacher ablation |

## 12. Primary metrics

Report:

- AP, AP50, AP75;
- APs, ARs;
- small-object false negatives;
- reflective false positives per image;
- white/transparent real-target recall;
- training overhead;
- inference latency;
- parameter count and FLOPs.

The method is not successful if reflective FP decreases by suppressing real bright targets.

## 13. Inference invariance check

With `zvc_enable=false` and during `model.eval()`:

- no teacher object should be constructed unless explicitly needed by the training process;
- no crop tensor should be generated;
- no ZVC module should appear in the forward graph;
- exported parameters and FLOPs should equal C1;
- validation/predict/export calls should follow the original C1 path.

## 14. Implementation package

- `modules/zoom_view_correction.py`: tested reference helpers;
- `tests/test_zoom_view_correction.py`: geometry, reliability, gradient, and hard-negative guards;
- `configs/hyp-c1-zvc.yaml`: conservative defaults;
- `CODEX_ZVC_IMPLEMENTATION_PROMPT.md`: actual-repository integration contract;
- `scripts/run_zvc_experiment_matrix.md`: controlled execution sequence.
