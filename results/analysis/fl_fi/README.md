# Reporting directional classification errors

These tables isolate two ways a classifier can confuse battles (`B`) with violence against civilians (`V`). **False Legitimation (FL)** is a gold `V` event assigned `B`; **False Illegitimation (FI)** is a gold `B` event assigned `V`. The terminology encodes the study's operational distinction between error directions, not a legal determination about an event.

## Build a report from predictions

With the repository environment active, run from the project root:

```bash
python -m pip install statsmodels
python results/analysis/fl_fi/generate_fl_fi_analysis.py \
  --country cmr --strategy zero_shot --sample-size 1000
```

`statsmodels` is an additional dependency of this generator and is absent from the root requirements. Inputs are per-model `ollama_results_*_acled_*_actors.csv` files beneath the selected condition in `results/`. Calculations read `true_label`, `pred_label`, and `actor_norm`; short label codes are required. Keep one matching CSV per model directory, since the generator selects the first match.

For few-shot reports, specify each demonstration count:

```bash
python results/analysis/fl_fi/generate_fl_fi_analysis.py \
  --country nga --strategy few_shot --sample-size 1000 --shots 1 3 5
```

Use `--actors Police Military` to choose actor-name filters and `--models mistral_7b gemma3_4b` to select model directories. Actor filters are case-insensitive regular-expression matches and may overlap. Each model's `Total` row always covers all input rows.

Reports are written beneath this directory's `cmr/` or `nga/` folder as `fl_fi_analysis_{strategy}.csv`, with a shot-count suffix for few-shot conditions. Supply `--output-dir` to retain separate reports for different sample sizes; the default filenames do not encode sample size.

## Interpret the columns

| Quantity | Calculation | Meaning |
| --- | --- | --- |
| `FL %` | `100 × FL Count / True V` | Fraction of civilian-violence events assigned battle. |
| `FI %` | `100 × FI Count / True B` | Fraction of battles assigned civilian violence. |
| `ΔLB` | `FI % − FL %` | Difference in percentage points. |
| `NBS` | `(FI % − FL %) / (FI % + FL %)` | Direction score from −1 (FL) to +1 (FI). |
| `RR` | `FI % / FL %` | Ratio of the two class-conditioned error rates. |

Total rows add Wilson 95% intervals and a two-sided binomial test of equal **error-direction counts**. This test does not establish equality of the class-conditioned rates when `True V` and `True B` differ. Read scores with their denominators: missing gold classes produce zero rates; no directional errors produce NBS zero and RR one; FI-only errors produce infinite RR.

Two implementation limitations affect this generator: a total with zero directional errors raises a type error in the significance calculation, and p-values at least `0.999` are rendered incorrectly as `<0.001`. Consult [generate_fl_fi_analysis.py](generate_fl_fi_analysis.py) when inspecting these cases.

## Relate these tables to other outputs

The pipeline's `fl_fi_by_model.csv` and `harm_metrics_detailed.csv` count `V → {B,S}` and `{B,S} → V`, including strategic developments (`S`). They also read the calibrated evaluation subset. This standalone generator uses only `V ↔ B` and reads the per-model prediction files, so its counts and denominators need not match the pipeline reports.

The library offers [cross-condition](../../../lib/analysis/generate_fl_fi_summary.py) and [per-actor](../../../lib/analysis/generate_per_actor_fl_fi.py) summary helpers. These assume 1,000-event conditions and the four vanilla model directories, and use coarse binomial p-value approximations. See the [analysis guide](../../../lib/README.md) for the broader evaluation workflow.
