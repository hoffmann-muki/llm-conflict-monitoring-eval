# Are LLMs Ready for Conflict Monitoring?

This is the companion repository for our arXiv preprint:

**Are LLMs Ready for Conflict Monitoring? Empirical Evidence from West Africa**

Hoffmann Muki and Olukunle Owolabi, 2026

arXiv: [2605.04177](https://arxiv.org/abs/2605.04177)

DOI: [10.48550/arXiv.2605.04177](https://doi.org/10.48550/arXiv.2605.04177)

The repository evaluates whether large language models can be used safely for conflict-event monitoring in West Africa. It measures event-classification accuracy, calibration, actor-based fairness, normative error direction, lexical robustness, and error-trace behavior for open-weight LLMs and domain-adapted baselines on ACLED data from Nigeria and Cameroon.

Our main finding is cautionary: current open-weight models are not ready for unsupervised conflict-monitoring deployment. We find statistically significant false-illegitimation bias in vanilla open-weight models, near-directional neutrality after domain adaptation, persistent actor-based selection bias even in adapted models, and strong sensitivity to geography-specific lexical framing.

This codebase supports local Ollama models, local Hugging Face causal LMs, ConfliBERT-style sequence classifiers, fine-tuned baselines, calibration, fairness metrics, harm metrics, counterfactual perturbation analysis, and FL/FI reporting.

## Paper-To-Code Map

| Paper concept | Repository implementation |
| --- | --- |
| Vanilla open-weight LLMs | Ollama/HF inference over `llama3.2:3b`, `mistral:7b`, `gemma3:4b`, and `olmo2:7b`. |
| AfroConfliBERT | Fine-tuned ConfliBERT baseline under `experiments/pipelines/conflibert/` and `models/conflibert_finetuned_*`. |
| AfroConfliLLAMA | Fine-tuned local causal LM / small-LLM baseline under `experiments/pipelines/ollama/` and `models/small_llm_merged_*`. |
| False Legitimization and False Illegitimization | `lib.analysis.harm`, `results/analysis/fl_fi/`, and `fl_fi_by_model.csv`. |
| Actor-based selection bias | `lib.analysis.metrics.compute_fairness_metrics` and `fairness_metrics_acled_{country}_actors.csv`. |
| Lexical robustness and perturbation sensitivity | `lib.analysis.counterfactual`, `word_impacts.csv`, and counterfactual figures. |
| Rationale and attribution tracing | `lib.analysis.error_trace` and `error_trace_report.json`. |

## Headline Findings From The Paper

We designed this repository to reproduce and extend the following findings from our paper:

- Vanilla open-weight models show statistically significant false-illegitimation bias; we report Gemma 3 4B misclassifying 18.29% of legitimate battles as civilian-targeted violence while making zero false-legitimation errors.
- Domain-adapted baselines, which we refer to as AfroConfliBERT and AfroConfliLLAMA, substantially reduce directional FL/FI harm and are close to directional neutrality.
- Domain adaptation does not remove actor-based selection bias; in Nigeria, state actors are legitimized 36.5% more often than non-state actors in comparable tactical contexts.
- Open-weight outputs are fragile to geography-specific lexical framing, with flip rates up to 66.7% in Cameroon and 34.2% in Nigeria for delegitimizing phrases.
- Error-trace profiling suggests that models can mask normative bias with unfaithful rationales, motivating robustness tests and human-in-the-loop review.

Terminology note: in the paper, we use "False Illegitimation" and "False Legitimation"; several repository files use the longer engineering labels `false_illegitimization` and `false_legitimization` for the same concepts.

The active task is ACLED event-type classification for Cameroon (`cmr`) and Nigeria (`nga`) using six labels:

| Code | ACLED event type |
| --- | --- |
| `V` | Violence against civilians |
| `B` | Battles |
| `E` | Explosions/Remote violence |
| `P` | Protests |
| `R` | Riots |
| `S` | Strategic developments |

## Repository Scope

This repository is intended for research replication, auditing, and extension. It is not a production conflict-monitoring system. We argue for human-in-the-loop oversight, adversarial robustness evaluation, and fairness-aware adaptation before models are used in high-stakes humanitarian or security workflows.

## What Is Here

| Path | Purpose |
| --- | --- |
| `experiments/` | Runnable pipelines, prompting strategies, split builders, fine-tuning scripts, and shell orchestrators. |
| `lib/` | Shared inference clients, data helpers, aggregation helpers, calibration, metrics, harm, counterfactual, and visualization modules. |
| `datasets/` | Country extracts and reusable balanced actor samples created from the ACLED source CSV. |
| `data/processed/splits/` | Leak-safe train/dev/test split bundles for supervised baselines. |
| `results/` | Prompt-experiment results, baseline results, figures, tables, and FL/FI summaries. |
| `scripts/` | Standalone utility scripts for normalizing ConfliBERT outputs and aggregating SPD. |
| `tests/` | Lightweight smoke tests for helpers and counterfactual components. |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Ollama-backed inference, start the Ollama daemon and make sure the model names in `lib/core/constants.py` are available:

```bash
ollama list
```

For ConfliBERT, download the base model once:

```bash
python experiments/pipelines/conflibert/download_conflibert_model.py \
  --out-dir models/conflibert
```

The main raw source path is configured as `datasets/Africa_lagged_data_up_to-2024-10-24.csv`.

## Main Prompting Experiment

The canonical prompt-experiment runner is:

```bash
COUNTRY=cmr SAMPLE_SIZE=1000 STRATEGY=zero_shot \
  ./experiments/scripts/run_full_analysis.sh
```

The script runs:

1. Per-model inference into model subdirectories.
2. Aggregation into `ollama_results_acled_{country}_actors.csv`.
3. Calibration, metrics, fairness, thresholds, harm, and per-class reports.
4. Counterfactual perturbation analysis, word-impact aggregation, visualizations, and error tracing.
5. A terminal summary of generated artifacts.

Common variants:

```bash
# Few-shot prompting with three examples per category.
COUNTRY=nga SAMPLE_SIZE=1000 STRATEGY=few_shot NUM_EXAMPLES=3 \
  ./experiments/scripts/run_full_analysis.sh

# Run a model subset.
COUNTRY=cmr SAMPLE_SIZE=1000 INFERENCE_MODELS="mistral:7b,llama3.2:3b" \
  ./experiments/scripts/run_full_analysis.sh

# Re-analyze existing per-model predictions.
COUNTRY=cmr SAMPLE_SIZE=1000 SKIP_INFERENCE=true \
  ./experiments/scripts/run_full_analysis.sh

# Skip the counterfactual phase.
COUNTRY=cmr SAMPLE_SIZE=1000 SKIP_COUNTERFACTUAL=true \
  ./experiments/scripts/run_full_analysis.sh
```

Prompt-experiment outputs use this layout:

```text
results/{country}/{strategy}/{sample_size}/
results/{country}/few_shot/{sample_size}/{num_examples}/
```

Each model writes a per-model CSV under a slugged subdirectory, for example:

```text
results/nga/zero_shot/1000/mistral_7b/ollama_results_mistral-7b_acled_nga_actors.csv
```

The shared sample is created once per country and sample size, then reused across models and strategies:

```text
datasets/{country}/state_actor_sample_{country}_{sample_size}.csv
```

## Inference Backends

The current pipeline can run three backend types:

| Backend | How it is selected | Notes |
| --- | --- | --- |
| Ollama chat API | Default for normal model names in `INFERENCE_MODELS`. | Uses structured JSON output through `lib/inference/ollama_client.py`. |
| Local HF causal LM | Add the model name to `HF_INFERENCE_MODELS` and provide `HF_MODEL_PATH` or `HF_MODEL_PATH_MAP`. | Used for merged fine-tuned small LMs without registering them in Ollama. |
| ConfliBERT sequence classifier | Model names starting with `conflibert` inside the Ollama pipeline, or the dedicated ConfliBERT scripts. | Dedicated baseline scripts are preferred for supervised experiments. |

HF-backed example:

```bash
COUNTRY=nga SAMPLE_SIZE=1000 STRATEGY=zero_shot \
INFERENCE_MODELS="acled-small-llm-ft:v1,mistral:7b" \
HF_INFERENCE_MODELS="acled-small-llm-ft:v1" \
HF_MODEL_PATH_MAP="acled-small-llm-ft:v1=models/small_llm_merged_acled_v1_seed42" \
  ./experiments/scripts/run_full_analysis.sh
```

## Fine-Tuned Baselines

Supervised baselines use leak-safe split bundles under `data/processed/splits/{split_version}/` and write comparison-ready results under `results/baselines/{split_version}/`.

```bash
SPLIT_VERSION=acled_v1 ./experiments/scripts/run_finetuned_baselines.sh
```

By default, this builds splits and fine-tunes/evaluates ConfliBERT. Small-LLM LoRA SFT is optional:

```bash
SPLIT_VERSION=acled_v1 RUN_SMALL_LLM=true \
SMALL_LLM_BASE_MODEL=models/Llama-3.2-3B \
  ./experiments/scripts/run_finetuned_baselines.sh
```

Important outputs:

| Path | Description |
| --- | --- |
| `data/processed/splits/{split_version}/manifest.json` | Split metadata, counts, and paths. |
| `results/baselines/{split_version}/conflibert/` | ConfliBERT held-out prediction CSVs. |
| `results/baselines/{split_version}/small_llm/` | Small-LLM held-out prediction CSVs. |
| `results/baselines/{split_version}/baseline_core_metrics.csv` | Accuracy, macro F1, and per-class F1 by model/country. |
| `results/baselines/{split_version}/baseline_fairness_metrics.csv` | Fairness metrics where computable. |
| `results/baselines/{split_version}/baseline_harm_metrics.csv` | FL/FI harm rates. |
| `results/baselines/{split_version}/baseline_publication_table.csv` | Merged reviewer-facing table. |

## Analysis Outputs

The full prompt pipeline can produce:

| File | Description |
| --- | --- |
| `ollama_results_acled_{country}_actors.csv` | Aggregated raw predictions. |
| `ollama_results_calibrated.csv` | Calibrated predictions. |
| `calibration_brier_scores.csv` and `reliability_diagrams.png` | Calibration diagnostics. |
| `metrics_acled_{country}_actors.csv` | Classification metrics. |
| `fairness_metrics_acled_{country}_actors.csv` | SPD and equalized-odds style metrics. |
| `selected_thresholds.json` and `selected_thresholds_per_class.csv` | Per-class decision thresholds. |
| `harm_metrics_detailed.csv` and `fl_fi_by_model.csv` | False legitimization and false illegitimization metrics. |
| `per_class_report.csv` and `per_class_metrics.png` | Per-class performance summaries. |
| `top_disagreements.csv` and `top_disagreements_table.png` | High-confidence disagreements and ambiguity annotations. |
| `error_cases_false_legitimization.csv` | Sampled `V -> B` cases. |
| `error_cases_false_illegitimization.csv` | Sampled `B -> V` cases. |
| `counterfactual_analysis_*.json` | Counterfactual perturbation results. |
| `counterfactual_analysis_summary.csv` | Counterfactual summary table. |
| `word_impacts.csv` and word-impact figures | Word-level perturbation impact summaries. |
| `error_trace_report.json` and `error_trace_summary.csv` | Rationale-flip concordance and attribution traces. |

Cross-model comparison can be run after prediction files exist:

```bash
COUNTRY=cmr STRATEGY=zero_shot SAMPLE_SIZE=1000 \
  python -m lib.analysis.compare_all_models
```

It writes `results/{country}/{strategy}/{sample_size}/comparison/` with combined metrics, fairness, harm tables, and comparison figures.

## FL/FI Summary Tables

Standalone FL/FI tables live under `results/analysis/fl_fi/`:

```bash
python results/analysis/fl_fi/generate_fl_fi_analysis.py \
  --country cmr --strategy zero_shot --sample-size 1000

python results/analysis/fl_fi/generate_fl_fi_analysis.py \
  --country cmr --strategy few_shot --sample-size 1000 --shots 1 3 5
```

See `results/analysis/fl_fi/README.md` for metric definitions and output format.

## Utility Scripts

```bash
# Aggregate SPD across result folders and create country/strategy plots.
python scripts/aggregate_spd_and_plot.py --countries cmr nga --strategy zero_shot

# Normalize legacy ConfliBERT labels in result CSVs. Creates .bak files first.
python scripts/normalize_conflibert_results.py --root results
```

## Testing

```bash
PYTHONPATH=. python tests/test_generic_pipeline.py

# Optional, if pytest is installed in your environment.
python -m pytest tests/ -v
```

`tests/test_counterfactual.py` expects a sample file under `datasets/nga/`; run a pipeline first if no sample exists.

## More Documentation

- `experiments/README.md` describes runnable pipelines and environment variables.
- `experiments/prompting_strategies/README.md` describes prompt strategy classes and registration.
- `lib/README.md` describes reusable modules and output conventions.
- `tests/README.md` describes the current smoke-test suite.
- `results/analysis/fl_fi/README.md` documents FL/FI reporting.

## Citation

If you use this repository, please cite our paper:

```bibtex
@misc{muki2026llmsconflictmonitoring,
  title = {Are LLMs Ready for Conflict Monitoring? Empirical Evidence from West Africa},
  author = {Muki, Hoffmann and Owolabi, Olukunle},
  year = {2026},
  eprint = {2605.04177},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  doi = {10.48550/arXiv.2605.04177},
  url = {https://arxiv.org/abs/2605.04177}
}
```
