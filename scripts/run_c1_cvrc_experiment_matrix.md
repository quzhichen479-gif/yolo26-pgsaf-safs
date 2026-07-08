# C1-CVRC experiment matrix

This file tells Codex exactly what to run after implementation. Keep C1 as the control and do not change the validation protocol.

## 0. Required controls

```text
E0: C1 baseline
E1: C1 + ordinary SAHI, if already available
E2: C1 + RAS selective, if already available
E4-v2c: RAS-v2c full-image inference
E5-v2c: RAS-v2c + RAS selective inference
E7: RAS-FSR failed control, if already available
```

Do not retrain controls unless existing metrics are missing or incompatible.

## 1. Smoke tests

Before any training:

```bash
# Adapt command names to the real YOLO26 repo.
python tools/smoke_cvrc.py --model configs/yolo26n-c1-cvrc.yaml --cvrc-disabled
python tools/smoke_cvrc.py --model configs/yolo26n-c1-cvrc.yaml --one-batch
python tools/smoke_cvrc.py --test-coordinate-roundtrip
python tools/smoke_cvrc.py --empty-label-image
python tools/smoke_cvrc.py --amp
```

Pass conditions:

```text
CVRC disabled: outputs match C1 or are near-identical.
One batch: no shape/label mapping crash.
Coordinate roundtrip: < 1 px mapping error.
Empty-label image: no crash.
AMP: no NaN.
```

## 2. E9-A: GT crop warmup only

Purpose: verify training-time crop-view adaptation can improve full-image inference without two-pass inference.

```bash
python train.py \
  --model configs/yolo26n-c1-cvrc.yaml \
  --data <FloatingWaste-I.yaml> \
  --weights <C1_best.pt> \
  --epochs 60 \
  --lr0 0.001 \
  --cvrc.enabled true \
  --cvrc.mode gt_crop_warmup_only \
  --rgz_infer.enabled false \
  --name E9A_C1_CVRC_GTcrop_fullinfer
```

Required outputs:

```text
metrics_original_protocol.csv
metrics_per_class.csv
latency.csv
cvrc_crop_stats.csv
cvrc_rectification_stats.csv
vis_cvrc_crops/
```

Accept if APs/AP75 improve without AP50 collapse.

## 3. E9-B: selective crop training

Purpose: make crop training focus on the same regions RAS/RGZ would select.

```bash
python train.py \
  --model configs/yolo26n-c1-cvrc.yaml \
  --data <FloatingWaste-I.yaml> \
  --weights <C1_best.pt> \
  --epochs 60 \
  --lr0 0.001 \
  --cvrc.enabled true \
  --cvrc.mode selective_crop_train \
  --rgz_infer.enabled false \
  --name E9B_C1_CVRC_selective_fullinfer
```

Compare E9-B against E9-A and C1.

## 4. E9-C: crop-view rectification loss

Purpose: directly improve full-view tiny/small localization quality.

```bash
python train.py \
  --model configs/yolo26n-c1-cvrc.yaml \
  --data <FloatingWaste-I.yaml> \
  --weights <C1_best.pt> \
  --epochs 60 \
  --lr0 0.001 \
  --cvrc.enabled true \
  --cvrc.mode selective_crop_rectify \
  --cvrc.loss.lambda_crop 0.35 \
  --cvrc.loss.lambda_box_rect 0.20 \
  --cvrc.loss.lambda_quality_rect 0.10 \
  --cvrc.loss.lambda_spec_neg 0.05 \
  --rgz_infer.enabled false \
  --name E9C_C1_CVRC_rectify_fullinfer
```

Primary check:

```text
mean_iou_gain > 0 in cvrc_rectification_stats.csv
AP75 improves
APs improves
```

## 5. E9-D: optional ZASG

Run only if E9-C is stable.

```bash
python train.py \
  --model configs/yolo26n-c1-cvrc.yaml \
  --data <FloatingWaste-I.yaml> \
  --weights <C1_best.pt> \
  --epochs 60 \
  --lr0 0.001 \
  --cvrc.enabled true \
  --cvrc.mode selective_crop_rectify \
  --zasg.enabled true \
  --rgz_infer.enabled false \
  --name E9D_C1_CVRC_ZASG_fullinfer
```

Reject ZASG if it behaves like ordinary attention and increases water false positives.

## 6. E9-E: optional RGZ inference

Run only after E9-C or E9-D produces a good checkpoint.

```bash
for z in 4 6 8; do
  python val.py \
    --model runs/detect/E9C_C1_CVRC_rectify_fullinfer/weights/best.pt \
    --data <FloatingWaste-I.yaml> \
    --imgsz 640 \
    --rgz_infer.enabled true \
    --rgz_infer.max_zooms $z \
    --rgz_infer.fusion spec_wbf \
    --name E9E_C1_CVRC_RGZ_z${z}
done
```

Required outputs:

```text
selected_zooms.json
fusion_stats.csv
duplicate_stats.csv
latency.csv
metrics_original_protocol.csv
```

## 7. Main summary table

Create a final CSV/Markdown table with:

```text
method
train_mode
infer_mode
AP
AP50
AP75
APs
ARs
latency_ms_img
avg_zooms
duplicate_boxes
reflection_FP
wave_FP
small_FN
```

## 8. Acceptance thresholds

Full-image CVRC:

```text
AP >= C1 - 0.1
AP50 >= C1 - 0.5
AP75 >= C1 + 1.0 preferred
APs >= C1 + 1.5 preferred
latency close to C1
```

Optional RGZ:

```text
AP >= 43.0
APs >= 34.0
AP75 >= 35.5
latency < E5-v2c
duplicate_boxes < E5-v2c
```

## 9. Decision logic

```text
If E9-A fails:
  stop single-pass CVRC and prioritize two-pass C1-RGZ.
If E9-A improves APs/AP75:
  continue to E9-B/E9-C.
If E9-C improves AP75 but AP50 drops:
  lower lambda_box_rect and lambda_quality_rect by half.
If hard negatives reduce FP but hurt APs:
  lower lambda_spec_neg or reduce hard_negative_ratio.
If optional RGZ is accurate but too slow:
  reduce max_zooms and improve proposal NMS.
```
