# C1-CVRC design: crop-view rectified C1

> Project: FloatingWaste-I water-surface floating waste detection  
> Baseline: YOLO26-C1 / `yolo26-safrg-pspatial`  
> Goal: transfer RAS-style crop-view localization ability back into C1 while keeping default inference clean.

## 1. Motivation

C1 already provides a stable full-image single-pass baseline by using pseudo-specular prior guided spatial recalibration after YOLO26 neck outputs and before Detect.

RAS-v2c shows that image-space crop resize can improve tiny/small localization and APs, especially when combined with selective slicing. However, RAS-v2c full-image inference alone drops below C1, so RAS-v2c should not be treated as a direct full-image model improvement.

The next step is therefore:

```text
Use crop-view only as training-time correction / optional inference-time zoom,
not as a replacement for C1 full-image detection.
```

## 2. Core hypothesis

The current bottleneck is not mainly proposal discovery. C1 already produces many tiny/small candidates. The harder issue is:

```text
candidate exists -> box is poorly localized or low-quality -> AP75/APs suffer
```

CVRC targets this by using original-image crop resize to obtain a clearer crop-view prediction, then using the mapped-back crop prediction to correct full-view small boxes.

## 3. Architecture

```text
Input image
   |
   |-- Full-view C1 branch
   |      |-- P3/P4/P5 neck outputs
   |      |-- C1 SAFRG recalibration
   |      |-- Detect full-image predictions
   |
   |-- Reflectance-guided crop selector
   |      |-- GT tiny/small boxes during warmup
   |      |-- low-confidence small C1 candidates after warmup
   |      |-- high Eedge / low Rcore windows
   |      |-- capped hard negatives and sparse fallback
   |
   |-- Crop-view branch
   |      |-- crop original image
   |      |-- resize to training size
   |      |-- run C1 on crop
   |      |-- map boxes back to original coordinates
   |
   |-- Rectification losses
          |-- crop detection loss
          |-- box rectification loss
          |-- quality rectification loss
          |-- specular hard-negative loss
```

Default inference keeps only the full-view path.

## 4. Why original-image crop is required

The effective RAS path is:

```text
original image crop -> resize -> real small-object pixels become larger -> detector sees true local detail
```

The failed feature-SR path is:

```text
full image resized to 640 -> P3 stride-8 feature loses detail -> try to recover detail in feature space
```

CVRC must therefore crop from the original image or from the highest-resolution image tensor before feature extraction. Do not crop P3 and call it zoom.

## 5. Training schedule

### Stage E9-A: GT crop warmup

- Use GT tiny/small boxes to generate positive crops.
- Add capped hard-negative crops from high specular-core regions.
- Run full-image detection loss and crop detection loss.
- Default inference remains full image only.

Expected role: prove crop-view training does not hurt C1 and can transfer some localization benefit.

### Stage E9-B: selective crop training

- Add low-confidence small predictions from C1 full branch.
- Add reflectance-edge windows.
- Keep GT crops mixed in to avoid selector drift.

Expected role: make training match the regions that will matter at inference/diagnosis.

### Stage E9-C: crop-view rectification

- Match full-view small candidates and crop-view mapped predictions.
- Apply rectification only when crop-view is clearly better.
- Record IoU gain statistics.

Expected role: directly improve AP75 and small-object box quality.

### Stage E9-D: Zoom-Aware Spatial Gate, optional

- Add a residual-safe P3 gate.
- Gate target comes from selected crop regions and crop-view improved boxes.
- Do not enable if E9-A/B/C are not stable.

### Stage E9-E: optional RGZ inference

- Enable selected zoom crop inference with `max_zooms` in {4, 6, 8}.
- Use spec-aware fusion.
- This is accuracy-first, not the default.

## 6. Crop selector design

Candidate sources:

1. GT tiny/small boxes during warmup.
2. Low-confidence small C1 predictions.
3. Reflectance-edge windows: high Eedge and low Rcore.
4. Hard-negative windows: high Rcore and low GT overlap.
5. Sparse fallback windows, strictly capped.

Candidate score:

```text
score(region) =
  1.0 * low_conf_small_candidate
+ 0.8 * mean(Eedge)
- 1.2 * mean(Rcore)
+ 0.5 * box_uncertainty
+ 0.3 * context_score
- 0.5 * overlap_penalty
```

## 7. Loss design

```text
L_total = L_det_full
        + lambda_crop * L_det_crop
        + lambda_box_rect * L_box_rectify
        + lambda_quality_rect * L_quality_rectify
        + lambda_spec_neg * L_spec_hard_negative
```

Initial weights:

```yaml
lambda_crop: 0.35
lambda_box_rect: 0.20
lambda_quality_rect: 0.10
lambda_spec_neg: 0.05
```

Rectification gating:

- object is tiny/small;
- crop-view prediction maps back to same GT or candidate;
- crop-view IoU exceeds full-view IoU by at least 0.05;
- mapped crop box is not dominated by isolated Rcore.

## 8. Evaluation

Primary metrics:

```text
AP
AP50
AP75
APs
ARs
latency
```

Required diagnostics:

```text
cvrc_crop_stats.csv
cvrc_selector_stats.csv
cvrc_rectification_stats.csv
small_fn_diagnostics.csv
localization_partial_overlap.csv
fp_reason_stats.csv
```

## 9. Acceptance criteria

Single-pass CVRC is useful if:

```text
AP is not lower than C1 by more than 0.1
AP75 improves by at least +1.0 preferred
APs improves by at least +1.5 preferred
latency remains close to C1
AP50 does not drop by more than 0.5
reflection/wave false positives do not clearly increase
```

Strong success:

```text
AP > C1
AP75 >= C1 + 1.5
APs >= C1 + 2.0
single-pass inference retained
```

Optional RGZ is useful if:

```text
AP >= 43.0
APs >= 34.0
AP75 >= 35.5
latency < E5-v2c
duplicate boxes lower than E5-v2c
```

## 10. Stop conditions

Stop or revert if:

- AP50 drops by more than 1.0;
- APs drops;
- reflection/wave false positives increase without AP75/APs gain;
- crop branch causes full-image distribution forgetting;
- crop count grows into ordinary SAHI behavior;
- duplicate boxes become worse than E5-v2c in optional RGZ mode.
