# Codex implementation prompt: C1-CVRC / RGZ for YOLO26-C1

You are working in the actual YOLO26 water-surface floating-waste detection repository. Implement a controlled C1 extension based on crop-view rectification. This task is not a generic attention-module experiment. It must preserve the validated C1 baseline and use RAS-style image-space crop/resize only as training supervision or an optional two-pass inference path.

## 0. Validated experimental facts

Use these facts as hard constraints:

- C1 (`yolo26-safrg-pspatial`) is the strongest and most stable full-image single-pass baseline.
- C1 inserts specular-aware feature recalibration after P3/P4/P5 neck outputs and before Detect.
- C1 does not change Detect, loss, assignment, dataset split, or the three-scale P3/P4/P5 detection interface.
- RAS-v2c improves performance only when combined with crop/sliced inference. It does not improve full-image inference by itself.
- The effective RAS mechanism is original-image crop resize, not feature-space super-resolution.
- Ordinary SAHI and teacher-free feature SR are not the main route.
- Oracle diagnostics indicate C1 already sees many tiny/small candidates; the bottleneck is mainly localization / box quality / ranking rather than pure proposal discovery.

Therefore the next implementation must optimize AP75 and APs by transferring crop-view localization knowledge back to C1.

## 1. Target method

Implement:

```text
C1-CVRC: Crop-View Rectified C1
```

Optional later inference mode:

```text
C1-RGZ: C1 with Reflectance-Guided Zoom inference
```

Core idea:

```text
Training:
full image -> C1 full branch -> normal YOLO detection loss
selected original-image crop -> resize -> C1 crop branch -> crop detection loss
crop predictions mapped back -> rectification loss for full-image small boxes

Inference default:
full image -> C1-CVRC full branch only

Inference optional:
full image -> C1 full branch -> selected zoom crops -> crop branch -> spec-aware fusion
```

The default inference path should remain full-image single-pass unless `--cvrc-zoom-infer` or an equivalent flag is explicitly enabled.

## 2. Non-negotiable constraints

Do not change:

- Detect head implementation.
- Detection loss formula except adding optional auxiliary CVRC losses outside the original loss.
- Label assignment logic.
- Dataset train/val split.
- Original C1 config file.
- Original C1 module behavior when CVRC is disabled.

Must keep:

- P3/P4/P5 Detect interface.
- C1 SAFRG / pseudo-specular prior path.
- C1 baseline reproducibility.

CVRC must be switchable by config/CLI flag.

## 3. Implementation phases

### Phase A: online GT crop warmup

Add a dataloader/trainer hook that, during training only, creates a few original-image crops around tiny/small ground-truth boxes.

Recommended defaults:

```yaml
cvrc:
  enabled: true
  warmup_epochs: 15
  crop_imgsz: 640
  max_positive_crops_per_image: 3
  max_total_crops_per_image: 4
  hard_negative_ratio: 0.10
  context_ratio: 0.05
  min_box_side_px: 2
  small_box_area_thr: 32*32
```

Warmup crop source:

- GT tiny/small boxes first.
- Add a small number of hard-negative crops from high `Rcore` and low label-overlap regions.
- Add sparse context crops only if total crops are fewer than the maximum.

### Phase B: selective crop training

After warmup, mix GT-guided crops with C1-guided selective crops.

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

Candidate sources:

1. Low-confidence small detections from the full branch.
2. Reflectance edge regions: high Eedge, not high Rcore.
3. Hard-negative regions: high Rcore with no GT overlap.
4. Sparse fallback windows, capped very small.

Important: do not make this dense-grid SAHI.

### Phase C: crop-view rectification loss

Forward crops through the same C1 model or a crop branch that shares most weights. Map crop predictions back to original-image coordinates.

Add auxiliary losses:

```text
L_total = L_det_full
        + lambda_crop * L_det_crop
        + lambda_box_rect * L_box_rectify
        + lambda_quality_rect * L_quality_rectify
        + lambda_spec_neg * L_spec_hard_negative
```

Recommended initial weights:

```yaml
lambda_crop: 0.35
lambda_box_rect: 0.20
lambda_quality_rect: 0.10
lambda_spec_neg: 0.05
```

Box rectification should apply only when:

- target is tiny/small;
- crop-view prediction matches the same GT or full-branch candidate;
- crop-view box has higher IoU to GT than the full-view box by a margin, e.g. +0.05;
- mapped crop prediction is not located inside a high isolated Rcore core.

### Phase D: Zoom-Aware Spatial Gate, optional

Add a small P3 gate supervised by selected crop regions, not a free attention block.

```text
P3_C1 -> ZASG -> P3_CVRC
Detect input = [P3_CVRC, P4_C1, P5_C1]
```

The gate mask target comes from selected crop windows / tiny GT boxes / crop-view improved predictions. Use residual-safe initialization:

```python
gamma = nn.Parameter(torch.zeros(1))
P3_out = P3_C1 + gamma * gate * refine(P3_C1)
```

Do not enable ZASG before Phase A/B/C works.

### Phase E: optional RGZ inference

Implement only after full-image CVRC training is stable.

```text
full C1 pass -> max_zooms in {4, 6, 8} -> crop pass -> map boxes back -> spec-aware WBF/NMS
```

RGZ is an optional accuracy-first mode, not the default.

## 4. Files to create in actual YOLO26 repo

Adapt the skeletons in this package into the real paths:

```text
ultralytics/nn/modules/cvrc_selector.py
ultralytics/nn/modules/crop_view_rectifier.py
ultralytics/nn/modules/zoom_aware_gate.py
ultralytics/engine/trainer_cvrc.py              # or the repo-equivalent trainer hook
ultralytics/utils/cvrc_fusion.py                # only for optional RGZ inference
configs/yolo26n-c1-cvrc.yaml
configs/yolo26n-c1-cvrc-zasg.yaml              # optional after CVRC works
configs/yolo26n-c1-rgz-infer.yaml              # optional inference config
```

Modify only if needed:

```text
ultralytics/nn/modules/__init__.py
ultralytics/nn/tasks.py or parser registry
train entry script / CLI arg parser
validation script for optional zoom inference
```

## 5. Required outputs

Every run must save:

```text
metrics_original_protocol.csv
metrics_per_class.csv
latency.csv
params_flops.csv
cvrc_crop_stats.csv
cvrc_rectification_stats.csv
cvrc_selector_stats.csv
small_fn_diagnostics.csv, if available
localization_partial_overlap.csv, if available
fp_reason_stats.csv, if available
vis_cvrc_crops/
vis_improved_worsened/
```

`cvrc_rectification_stats.csv` should include at least:

```text
image_id
num_full_small_candidates
num_crop_predictions
num_matched_rectified
mean_iou_full
mean_iou_crop_mapped
mean_iou_gain
mean_box_l1_delta
mean_rcore_inside
mean_eedge_inside
```

## 6. Experiment matrix

Baseline:

```text
E0: C1 baseline, unchanged
```

Main CVRC:

```text
E9-A: C1 + online GT crop training, full-image inference only
E9-B: C1 + selective crop training, full-image inference only
E9-C: C1 + selective crop training + rectification loss, full-image inference only
E9-D: C1 + CVRC + ZASG, full-image inference only
```

Optional RGZ:

```text
E9-E4: C1-CVRC + RGZ inference, max_zooms=4
E9-E6: C1-CVRC + RGZ inference, max_zooms=6
E9-E8: C1-CVRC + RGZ inference, max_zooms=8
```

Compare against:

```text
E0 C1 baseline
E1 ordinary SAHI
E2 C1 + RAS selective
E4-v2c full image
E5-v2c RAS selective
E6-v2c ordinary SAHI
E7 RAS-FSR failed control, if available
```

## 7. Acceptance criteria

For full-image inference CVRC:

```text
AP > E0 AP - 0.1
AP75 >= E0 AP75 + 1.0 preferred
APs >= E0 APs + 1.5 preferred
latency close to C1
AP50 must not drop by more than 0.5
reflection/wave FP must not increase obviously
```

Strong success:

```text
AP > E0
AP75 >= E0 + 1.5
APs >= E0 + 2.0
same single-pass inference style as C1
```

For optional RGZ inference:

```text
AP >= 43.0
APs >= 34.0
AP75 >= 35.5
latency < E5-v2c
duplicate boxes clearly lower than E5-v2c
```

## 8. Implementation warnings

- Do not silently train on only crop images. Full images must remain the main distribution.
- Do not let crop count exceed conservative RAS-v2c density.
- Do not make hard negatives dominate.
- Do not use ordinary dense SAHI grids.
- Do not claim success from AP only; inspect AP75, APs, small FN, localization partial overlaps, and FP reasons.
- If E9-A fails, stop single-pass CVRC and move to optional C1-RGZ two-pass inference.
- If E9-A improves AP75/APs, continue to E9-B/E9-C.

## 9. Smoke tests before training

Implement and run:

```text
1. CVRC disabled: output must be numerically identical or near-identical to C1.
2. One training batch with GT crops: no shape or label mapping errors.
3. Crop box mapping round trip: original -> crop -> original has < 1 px error.
4. Empty-label image: no crash, at most fallback/hard-negative crops.
5. No spec prior exposed: code falls back safely without breaking C1.
6. Mixed precision: no NaN in crop branch or rectification loss.
```

## 10. Final deliverable

Open a PR with:

- implementation code;
- configs;
- smoke-test script;
- experiment commands;
- README update explaining CVRC and optional RGZ;
- no modification to the original C1 config except copies.
