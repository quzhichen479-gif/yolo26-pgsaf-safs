# Y-series experiment matrix: BUDQ-YOLO26

This file defines the controlled Y-series experiment plan for YOLO26-C1 + BUDQ-YOLO26 Loss.

The Y-series must be run after fixing the C1 initialization issue observed in the RSQ cohort. Do not interpret single-run differences as causal loss effects.

## 1. Mandatory fixed initialization

Before Y0-Y5, implement one of these protocols:

### Option A: reset seed before model construction

Ensure all RNGs are reset before every `YOLO(C1_yaml)` construction.

### Option B: fixed transferred initialization

1. Construct YOLO26-C1 model.
2. Load YOLO26n transfer weights.
3. Save the resulting initialized model as `C1_init.pt`.
4. Train every Y-series variant from exactly this `C1_init.pt`.

Preferred option: **Option B**, because it is easier to audit.

## 2. Fixed training protocol

Keep identical across Y0-Y5:

```text
dataset split
image size
optimizer
base learning rate
epochs
batch size
augmentation
assignment
Detect head
evaluation protocol
validation confidence/NMS settings
```

Do not change DFL, because YOLO26 should remain DFL-free.

## 3. Y-series variants

| ID | Loss setting | Ranking | Purpose |
|---|---|---|---|
| Y0 | original YOLO26-C1 loss | off | fixed-init baseline |
| Y1 | NWD-only or CIoU+NWD | off | recheck L2 signal |
| Y2 | MPDIoU+NWD | off | recheck L3 signal |
| Y3 | UBR-box | off | boundary uncertainty box loss |
| Y4 | UBR-box + posneg ranking | on, no duplicate | low-confidence TP ranking |
| Y5 | UBR-box + posneg + duplicate ranking | full | FreeNMS duplicate ordering |

Recommended order:

```text
Y0 -> Y1 -> Y2 -> Y3 -> Y4 -> Y5
```

Stop early if Y3 fails clearly.

## 4. Suggested hyperparameter templates

### Y0: original C1 loss

```yaml
box_loss_type: original
budq_enable: false
budq_enable_rank: false
budq_enable_dup_rank: false
```

### Y1: NWD control

```yaml
box_loss_type: nwd
budq_enable: false
budq_nwd_c: 16.0
```

### Y2: MPDIoU+NWD control

```yaml
box_loss_type: mpdiou_nwd
budq_enable: false
budq_tau: 32.0
budq_nwd_c: 16.0
budq_lambda_mpd: 0.5
budq_lambda_nwd: 1.0
```

### Y3: UBR-box

```yaml
box_loss_type: budq_ubr
budq_enable: true
budq_enable_rank: false
budq_enable_dup_rank: false
budq_tau: 32.0
budq_nwd_c: 16.0
budq_rho: 0.10
budq_u_min: 1.0
budq_u_max: 4.0
budq_tight_warmup_epochs: 30
budq_lambda_cover: 1.0
budq_lambda_spill: 0.5
budq_lambda_nwd: 1.0
budq_lambda_mpd: 0.5
```

### Y4: UBR + positive/background ranking

```yaml
box_loss_type: budq_ubr
budq_enable: true
budq_enable_rank: true
budq_enable_dup_rank: false
budq_rank_margin: 0.10
budq_rank_topk_neg: 32
budq_pos_iou_thr: 0.50
budq_neg_iou_thr: 0.25
budq_lambda_rank_posneg: 0.2
budq_lambda_rank_dup: 0.0
```

### Y5: UBR + positive/background + duplicate ranking

```yaml
box_loss_type: budq_ubr
budq_enable: true
budq_enable_rank: true
budq_enable_dup_rank: true
budq_rank_margin: 0.10
budq_dup_margin: 0.05
budq_rank_topk_neg: 32
budq_pos_iou_thr: 0.50
budq_neg_iou_thr: 0.25
budq_lambda_rank_posneg: 0.2
budq_lambda_rank_dup: 0.1
```

## 5. Smoke tests

Before each full training run:

```text
one forward pass
one loss computation
one optimizer step
100-iteration mini-train if supported
```

Check:

```text
finite loss
no NaN / Inf
positive sample count normal
L_cover in reasonable range
L_spill non-negative
L_nwd non-negative
L_mpdiou non-negative
alpha_s_mean in [0,1]
beta_t in [0,1]
ranking loss does not dominate total loss
```

## 6. Required logs

Every run must save:

```text
loss_components.csv
metrics_original_protocol.csv
metrics_per_class.csv
small_fn_diagnostics.csv
localization_partial_overlap.csv
fp_reason_stats.csv
duplicate_stats.csv
low_conf_tp_stats.csv
latency.csv
params_flops.csv
visualizations/improved_cases/
visualizations/worsened_cases/
```

If a diagnostic script is unavailable, create a README in the run folder saying which diagnostic is unavailable.

## 7. Three-seed requirement

After smoke tests, run at least three seeds or three fixed-initialization repeats:

```text
seed 42
seed 43
seed 44
```

Report mean and standard deviation.

## 8. Comparison table template

| ID | AP | AP50 | AP75 | APs | ARs | Bottle AP | Carton AP | FP | FN | Low-conf FN | Poor-loc/duplicate FP | Reflection FP | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Y0 | | | | | | | | | | | | | |
| Y1 | | | | | | | | | | | | | |
| Y2 | | | | | | | | | | | | | |
| Y3 | | | | | | | | | | | | | |
| Y4 | | | | | | | | | | | | | |
| Y5 | | | | | | | | | | | | | |

## 9. Acceptance criteria

### Y3 accepted if

```text
AP75 improves over Y0/Y2
APs does not drop clearly
poor-localization FP decreases
FN does not increase sharply
```

### Y4 accepted if

```text
low-confidence FN decreases
APs or ARs improves
high-score background FP does not increase
precision does not collapse
```

### Y5 accepted if

```text
duplicate / poor-localization FP decreases
AP75 remains stable or improves
FreeNMS output duplicate count decreases
```

## 10. Rejection criteria

Reject a variant if:

```text
AP50 drops > 1.0
APs drops clearly
FN rises sharply
ranking loss increases duplicate FP
training becomes unstable
any DFL/reg_max path is introduced
```

## 11. Final report format

Use this exact structure:

```text
Best Y-series variant: Y?

Fixed-init protocol:
- C1_init.pt: yes/no
- seeds: ...

Main metrics:
- ΔAP vs Y0: ...
- ΔAP75 vs Y0: ...
- ΔAPs vs Y0: ...
- ΔARs vs Y0: ...

Error audit:
- ΔFN: ...
- Δlow-conf FN: ...
- Δpoor-loc/duplicate FP: ...
- Δreflection FP: ...

Conclusion:
- NWD signal confirmed: yes/no
- MPDIoU+NWD confirmed: yes/no
- UBR boundary uncertainty useful: yes/no
- Ranking useful: yes/no
- Recommended next step: ...
```
