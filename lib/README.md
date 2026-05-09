# Library Modules

`lib/` contains the reusable analysis and infrastructure code for our paper **Are LLMs Ready for Conflict Monitoring? Empirical Evidence from West Africa** ([arXiv:2605.04177](https://arxiv.org/abs/2605.04177)). It provides data extraction, path conventions, inference clients, result aggregation, calibration, fairness metrics, harm analysis, counterfactual robustness analysis, error tracing, and visualizations.

The library is organized around our core empirical questions: whether models misclassify legitimate battles as civilian-targeted violence, whether state actors receive systematically different treatment than non-state actors, whether lexical framing can flip predictions, and whether reasoning traces faithfully reflect model behavior.

## Layout

```text
lib/
|-- analysis/          # Calibration, metrics, harm, counterfactual, error tracing, tables, plots
|-- core/              # Constants, paths, strategy factory, result aggregation, shared metric helpers
|-- data_preparation/  # ACLED country extraction, actor normalization, sampling
`-- inference/         # Ollama, HF causal LM, and ConfliBERT inference helpers
```

## Core Conventions

```python
from lib.core.data_helpers import (
    setup_country_environment,
    paths_for_country,
    get_strategy,
    get_sample_size,
    get_num_examples,
    model_name_to_dir_slug,
)
```

Important path behavior:

| Scenario | Results directory |
| --- | --- |
| Zero-shot or explainable | `results/{country}/{strategy}/{sample_size}` |
| Few-shot with `NUM_EXAMPLES=3` | `results/{country}/few_shot/{sample_size}/3` |
| `RESULTS_DIR` environment variable set | Used as an explicit override |

Shared samples use:

```text
datasets/{country}/state_actor_sample_{country}_{sample_size}.csv
```

Model subdirectories use underscore slugs, while per-model filenames use colon-to-hyphen slugs:

```text
results/nga/zero_shot/1000/mistral_7b/ollama_results_mistral-7b_acled_nga_actors.csv
```

## Constants

```python
from lib.core.constants import (
    LABEL_MAP,
    EVENT_CLASSES_FULL,
    CSV_SRC,
    WORKING_MODELS,
    LOCAL_BASE_MODELS,
    COUNTRY_NAMES,
)
```

The current country codes are `cmr` for Cameroon and `nga` for Nigeria. The current working model list includes:

```text
llama3.2:3b
mistral:7b
gemma3:4b
olmo2:7b
acled-small-llm-ft:v1
```

## Strategy Factory

```python
from lib.core.strategy_helpers import get_strategy, STRATEGY_REGISTRY

strategy = get_strategy("few_shot", num_examples=3)
prompt = strategy.make_prompt("Event notes...")
schema = strategy.get_schema()
```

Registered strategies are `zero_shot`, `few_shot`, and `explainable`.

## Result Aggregation

```python
from lib.core.result_aggregator import (
    aggregate_model_results,
    write_combined_results,
    get_per_model_result_path,
    list_per_model_files,
)
```

CLI:

```bash
python -m lib.core.result_aggregator \
  --country cmr --strategy zero_shot --sample-size 1000

python -m lib.core.result_aggregator \
  --country cmr --strategy few_shot --sample-size 1000 --num-examples 3 \
  --models "mistral:7b,llama3.2:3b"
```

The aggregator scans `results_dir/*/ollama_results_*_acled_{country}_actors.csv`, deduplicates by `(model, event_id)`, and writes:

```text
ollama_results_acled_{country}_actors.csv
```

## Data Preparation

```python
from lib.data_preparation import (
    extract_country_rows,
    get_actor_norm_series,
    extract_state_actor,
    build_stratified_sample,
    build_balanced_actor_sample,
)
```

Prompt experiments currently use balanced actor sampling to support fairness analysis:

```python
sample = build_balanced_actor_sample(
    df,
    n_total=1000,
    balance_ratio=0.5,
    event_types=EVENT_CLASSES_FULL,
    actor_code_col="INTER1",
    min_per_cell=10,
    label_map=LABEL_MAP,
    random_state=42,
)
```

Output samples include the canonical fields used downstream: `event_id_cnty`, `notes`, `gold_label`, `gold_label_full`, `actor_type`, and `actor_norm` when available.

## Inference Clients

### Ollama

```python
from lib.inference.ollama_client import run_ollama_structured

resp = run_ollama_structured(
    "mistral:7b",
    prompt=strategy.make_prompt("Event notes..."),
    system_msg=strategy.get_system_message(),
    schema=strategy.get_schema(),
)
```

Ollama inference posts to `http://localhost:11434/api/chat`, asks for structured JSON, and normalizes labels to `V/B/E/P/R/S`.

### Local HF Causal LM

`lib.inference.hf_causal_client` supports direct inference from merged local HF checkpoints. The main pipeline enables it with:

```bash
HF_INFERENCE_MODELS="acled-small-llm-ft:v1"
HF_MODEL_PATH="models/small_llm_merged_acled_v1_seed42"
```

For multiple HF-backed names, use:

```bash
HF_MODEL_PATH_MAP="model_a=/path/a,model_b=/path/b"
```

### ConfliBERT

`lib.inference.conflibert_client` provides single-text inference helpers. The dedicated ConfliBERT experiment and baseline scripts are in `experiments/pipelines/conflibert/`.

## Analysis Modules

Most analysis modules can be run with `python -m` and accept `--country`, `--strategy`, `--sample-size`, and, for few-shot runs, `--num-examples`. Together, these modules produce the accuracy, fairness, harm, calibration, robustness, and error-trace artifacts used in our paper.

```bash
python -m lib.analysis.calibration --country cmr --strategy zero_shot --sample-size 1000
python -m lib.analysis.metrics --country cmr --strategy zero_shot --sample-size 1000
python -m lib.analysis.thresholds --country cmr --strategy zero_shot --sample-size 1000
python -m lib.analysis.harm --country cmr --strategy zero_shot --sample-size 1000
python -m lib.analysis.per_class_metrics --country cmr --strategy zero_shot --sample-size 1000
python -m lib.analysis.visualize_reports
```

Counterfactual and trace modules:

```bash
python -m lib.analysis.counterfactual \
  --country cmr --strategy zero_shot --sample-size 1000 \
  --models "mistral:7b,llama3.2:3b" --events 20

python -m lib.analysis.visualize_counterfactual \
  --input results/cmr/zero_shot/1000/counterfactual_analysis_mistral-7b.json

python -m lib.analysis.aggregate_word_impacts_from_counterfactuals \
  --models "mistral:7b,llama3.2:3b"

python lib/analysis/error_trace.py \
  --counterfactual-json results/cmr/zero_shot/1000/counterfactual_analysis_mistral-7b.json
```

Comparison modules:

```bash
python -m lib.analysis.compare_all_models

python -m lib.analysis.compare_models \
  --country cmr --strategy zero_shot --sample-size 1000 \
  --family gemma3 --sizes 4b

python -m lib.analysis.compare_finetuned_baselines \
  --split-version acled_v1 --results-root results/baselines
```

Additional analysis helpers:

| Module | Purpose |
| --- | --- |
| `lib.analysis.auto_annotate` | Heuristic feature extraction for source provenance, casualty cues, passive voice, verb intensity, and actor ambiguity. |
| `lib.analysis.event_ambiguity` | Event Ambiguity Score utilities used to annotate high-disagreement events. |
| `lib.analysis.generate_fl_fi_summary` | Country-level FL/FI summary across `zero_shot` and selected few-shot conditions. |
| `lib.analysis.generate_per_actor_fl_fi` | Per-actor FL/FI tables for fixed conditions. |
| `lib.analysis.visualize_word_impacts_by_perturbation` | Plots word-impact summaries by perturbation type. |

## Output Files

Prompt-experiment outputs are written to the active results directory.

| Group | Files |
| --- | --- |
| Inference | `{model_slug}/ollama_results_{model-file-slug}_acled_{country}_actors.csv`, `ollama_results_acled_{country}_actors.csv` |
| Calibration | `ollama_results_calibrated.csv`, `calibration_brier_scores.csv`, `isotonic_mappings.json`, `reliability_diagrams.png` |
| Metrics | `metrics_acled_{country}_actors.csv`, `fairness_metrics_acled_{country}_actors.csv`, `confusion_matrices_acled_{country}_actors.json` |
| Thresholds | `selected_thresholds.json`, `selected_thresholds_per_class.csv`, `metrics_thresholds_calibrated.csv`, `accuracy_vs_coverage.png` |
| Harm | `harm_metrics_detailed.csv`, `fl_fi_by_model.csv` |
| Error analysis | `per_class_report.csv`, `top_disagreements.csv`, `error_cases_false_legitimization.csv`, `error_cases_false_illegitimization.csv`, `error_correlations_acled_{country}_actors.csv` |
| Counterfactual | `counterfactual_analysis_*.json`, `counterfactual_analysis_summary.csv`, per-model `counterfactual_report.txt`, `summary_table.png`, `flip_rates_by_perturbation.png` |
| Word impacts | `word_impacts.csv`, `fig_word_impact_scatter_by_perturbation.png`, `fig_word_impact_bars_by_perturbation.png` |
| Error trace | `error_trace_report.json`, `error_trace_summary.csv` |
| Cross-model comparison | `comparison/all_models_metrics.csv`, `comparison/all_models_fairness.csv`, `comparison/all_models_harm.csv`, `comparison/*.png` |

Fine-tuned baseline outputs are written to:

```text
results/baselines/{split_version}/
```

with `baseline_core_metrics.csv`, `baseline_fairness_metrics.csv`, `baseline_harm_metrics.csv`, and `baseline_publication_table.csv`.
