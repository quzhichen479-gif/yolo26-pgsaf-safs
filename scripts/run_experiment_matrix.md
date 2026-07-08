# Recommended experiment matrix

Keep the exact same training protocol as C1 baseline unless the existing project has a fixed ablation protocol.

## Models

1. C1 baseline
2. C1 + PG-SAF
3. C1 + SAFSBlock
4. C1 + PG-SAF + SAFSBlock, only if 2 or 3 is positive

## Required outputs

- metrics_original_protocol.csv
- metrics_per_class.csv
- confusion / FP-FN diagnostic files if available
- small_fn_diagnostics.csv
- localization_partial_overlap.csv
- fp_reason_stats.csv with at least:
  - reflection_highlight_background
  - wave_ripple
  - foam
  - shadow/reflection
  - bank_clutter
- latency.csv
- params_flops.csv
- visualization samples:
  - improved small FN cases
  - worsened reflection/wave FP cases
  - P3/P4 fusion activation maps if available

## Success criteria

PG-SAF is useful only if:

- AP does not drop more than 0.2;
- APs improves by at least +0.5;
- AP75 improves or localization_partial_overlap decreases;
- wave/ripple and reflection FP do not increase clearly.

SAFSBlock is useful only if:

- APs or AP75 improves;
- reflection_highlight_background_FP or wave_ripple_FP decreases;
- latency increase is acceptable;
- no large AP50 collapse.

Stop conditions:

- AP50 drops by more than 1.0;
- APs drops;
- reflection/wave FP increases while AP gain is tiny;
- latency increase is not justified.
