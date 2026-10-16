# From predictions to evidence

The `lib` package supplies the common data, inference, and analysis code used by the [experiment runners](../experiments/README.md). Its central interface is an event-level prediction table: each backend writes the same identifiers and labels so the analysis can be repeated independently of model generation.

## Prediction tables

The prompt pipeline writes the following fields to CSV:

| Fields | Meaning |
| --- | --- |
| `model`, `event_id` | Model identity and source event identifier; together they identify a prediction. |
| `true_label`, `pred_label` | Gold and predicted short labels. Generation failures may appear as `ERROR`. |
| `pred_conf`, `logits` | Scalar confidence and a JSON-encoded class-score object. |
| `actor_norm`, `notes` | Actor description and event narrative for fairness and perturbation analysis. |
| `reasoning`, `latency_sec` | Returned rationale, when present, and inference time. |

The canonical label order is `V, B, E, P, R, S`, corresponding to violence against civilians, battles, explosions/remote violence, protests, riots, and strategic developments. The generative prompts request probabilities in the legacy `logits` field; classifier backends may produce different score representations. Consult the relevant [inference client](inference/) before interpreting scores across backends.

## Processing stages

```text
Country sample → per-model predictions → combined table → calibration subset
                                                           │
                                        performance · fairness · harm
                                                           │
                                           disagreement cases → perturbations → traces
```

[Aggregation](core/result_aggregator.py) collects per-model CSVs and retains the last occurrence of each `(model, event_id)` pair. [Calibration](analysis/calibration.py) fits isotonic mappings on 80% of valid predictions per model and writes the remaining 20% to `ollama_results_calibrated.csv`, using seed 42. With fewer than two valid rows, it reuses the available rows for both steps. Downstream reports that read this calibrated table describe that subset, rather than the entire input sample.

To run these stages directly, set the condition once in a shell:

```bash
export COUNTRY=cmr STRATEGY=zero_shot SAMPLE_SIZE=1000
python -m lib.core.result_aggregator
python -m lib.analysis.calibration
python -m lib.analysis.metrics
python -m lib.analysis.harm
```

For few-shot conditions, also export `NUM_EXAMPLES`. [data_helpers.py](core/data_helpers.py) resolves these settings into a results directory; its callers can override that directory with `RESULTS_DIR`. Use `--list-only` on the aggregator to inspect discovered files before combining them.

## Reports to inspect

The following files are relative to the active prompt-condition directory:

| Question | Output | Implementation in `analysis/` |
| --- | --- | --- |
| How well does the model classify events? | `metrics_acled_{country}_actors.csv`, `per_class_report.csv` | [metrics](analysis/metrics.py), [per_class_metrics](analysis/per_class_metrics.py) |
| Does confidence track correctness? | `calibration_brier_scores.csv`, `reliability_diagrams.png` | [calibration](analysis/calibration.py) |
| Do predictions differ by actor group? | `fairness_metrics_acled_{country}_actors.csv` | [metrics](analysis/metrics.py) |
| Which direction do harmful errors take? | `harm_metrics_detailed.csv`, `fl_fi_by_model.csv` | [harm](analysis/harm.py) |
| What happens when the model abstains? | `selected_thresholds.json`, `accuracy_vs_coverage.png` | [thresholds](analysis/thresholds.py) |
| Can lexical changes flip a prediction? | `counterfactual_analysis_*.json`, `word_impacts.csv` | [counterfactual](analysis/counterfactual.py), [word-impact aggregation](analysis/aggregate_word_impacts_from_counterfactuals.py) |
| How do rationales respond to flips? | `error_trace_report.json`, `error_trace_summary.csv` | [error_trace](analysis/error_trace.py) |

Fairness uses actor-name keywords to derive state/non-state groups. Its default Statistical Parity Difference is `P(pred=V | state) − P(pred=V | non-state)`; TPR and FPR differences use the same ordering. This grouping rule differs from the ACLED actor-code rule used to balance samples.

Pipeline harm counts `V → {B,S}` as FL and `{B,S} → V` as FI. The detailed rates divide by true `V` and true `{B,S}` counts. The [standalone FL/FI analysis](../results/analysis/fl_fi/README.md) uses the narrower `V ↔ B` definition.

## Where to make changes

Use [data_preparation/](data_preparation/) for extraction and sampling, [inference/](inference/) for backend behavior, [core/constants.py](core/constants.py) for labels and model names, and [analysis/](analysis/) for evaluation. Prompt text lives in [experiments/prompting_strategies/](../experiments/prompting_strategies/README.md). The full runner coordinates these components and handles model-dependent perturbation and tracing stages.
