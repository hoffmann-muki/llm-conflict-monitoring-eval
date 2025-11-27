# Library

Reusable library components for LLM state actor bias analysis.

## Structure

```
lib/
├── analysis/          # Metrics, calibration, fairness, counterfactual
├── core/              # Constants, helpers, result aggregation
├── data_preparation/  # Data extraction, normalization, sampling
└── inference/         # Ollama model inference
```

## Modules

### Data Preparation

Data extraction and sampling utilities:

```python
from lib.data_preparation import (
    extract_country_rows,
    get_actor_norm_series,
    extract_state_actor,
    build_stratified_sample
)
```

| Function | Description |
|----------|-------------|
| `extract_country_rows()` | Extract country-specific rows from ACLED data |
| `get_actor_norm_series()` | Normalize actor names |
| `extract_state_actor()` | Identify state actors |
| `build_stratified_sample()` | Create stratified samples with optional oversampling |

### Core

Constants, utilities, and result aggregation:

```python
from lib.core.constants import LABEL_MAP, EVENT_CLASSES_FULL, WORKING_MODELS
from lib.core.data_helpers import setup_country_environment, paths_for_country
from lib.core.metrics_helpers import aggregate_fl_fi, LEGIT, ILLEG
from lib.core.result_aggregator import aggregate_model_results, get_per_model_result_path
```

**Result Aggregator:**

Combines per-model inference files into a single aggregated file for cross-model analysis:

```python
from lib.core.result_aggregator import aggregate_model_results, model_name_to_slug

# Aggregate all per-model files
COUNTRY=cmr STRATEGY=zero_shot python -m lib.core.result_aggregator

# Programmatic usage
from lib.core.result_aggregator import get_per_model_result_path
path = get_per_model_result_path('cmr', 'mistral:7b', strategy='zero_shot')
# Returns: results/cmr/zero_shot/ollama_results_mistral-7b_acled_cmr_state_actors.csv

# Convert model name to filename-safe slug
slug = model_name_to_slug('llama3.1:8b')  # Returns: 'llama3.1-8b'
```

### Inference

Ollama model inference with structured JSON output:

```python
from lib.inference.ollama_client import run_ollama_structured
from experiments.prompting_strategies import ZeroShotStrategy

strategy = ZeroShotStrategy()
prompt = strategy.make_prompt('Event description')
system_msg = strategy.get_system_message()
result = run_ollama_structured('gemma:2b', prompt, system_msg)
# Returns: {"label": "V", "confidence": 0.9}
```

**Note:** `run_ollama_structured()` requires an explicit `prompt` parameter. Use prompting strategies to generate prompts.

### Analysis

All analysis modules are runnable as Python modules:

```bash
# Classification metrics and fairness
COUNTRY=cmr STRATEGY=zero_shot python -m lib.analysis.metrics

# Calibration (isotonic + temperature scaling)
COUNTRY=cmr STRATEGY=zero_shot python -m lib.analysis.calibration

# Harm analysis (FL/FI rates)
COUNTRY=cmr STRATEGY=zero_shot python -m lib.analysis.harm

# Per-class metrics and error sampling
COUNTRY=cmr STRATEGY=zero_shot python -m lib.analysis.per_class_metrics

# Counterfactual perturbation testing
COUNTRY=cmr STRATEGY=zero_shot python -m lib.analysis.counterfactual --events 20

# Or specify models explicitly
COUNTRY=cmr STRATEGY=zero_shot python -m lib.analysis.counterfactual \
  --models llama3.2,mistral:7b --events 20

# Use percentage of available disagreements
COUNTRY=cmr STRATEGY=zero_shot python -m lib.analysis.counterfactual --top-percent 10

# Visualizations
COUNTRY=cmr STRATEGY=zero_shot python -m lib.analysis.visualize_reports

# Model size comparisons
COUNTRY=cmr STRATEGY=zero_shot python -m lib.analysis.compare_models \
  --family gemma --sizes 2b,7b

# Decision thresholds
COUNTRY=cmr STRATEGY=zero_shot python -m lib.analysis.thresholds
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `COUNTRY` | Country code (cmr, nga) | cmr |
| `STRATEGY` | Prompting strategy (zero_shot, few_shot, explainable) | zero_shot |
| `RESULTS_DIR` | Results directory path | Auto-detected |

## Directory Structure

Results are organized by country and strategy:

```
results/
├── cmr/
│   ├── zero_shot/
│   │   ├── ollama_results_mistral-7b_acled_cmr_state_actors.csv
│   │   ├── ollama_results_acled_cmr_state_actors.csv
│   │   ├── ollama_results_calibrated.csv
│   │   ├── metrics_acled_cmr_state_actors.csv
│   │   └── ...
│   └── few_shot/
│       └── ...
└── nga/
    ├── zero_shot/
    └── few_shot/
```

## Output Files

Each analysis module writes to `results/<country>/<strategy>/`:

### Inference
| File | Description |
|------|-------------|
| `ollama_results_{model}_acled_{country}_state_actors.csv` | Per-model results |
| `ollama_results_acled_{country}_state_actors.csv` | Combined results |

### Calibration
| File | Description |
|------|-------------|
| `ollama_results_calibrated.csv` | Calibrated predictions |
| `calibration_brier_scores.csv` | Brier scores |
| `isotonic_mappings.json` | Isotonic calibration mappings |
| `reliability_diagrams.png` | Calibration visualization |
| `accuracy_vs_coverage.png` | Threshold analysis |

### Metrics
| File | Description |
|------|-------------|
| `metrics_acled_{country}_state_actors.csv` | Classification metrics |
| `fairness_metrics_acled_{country}_state_actors.csv` | SPD, Equalized Odds |
| `confusion_matrices_acled_{country}_state_actors.json` | Confusion matrices |
| `error_correlations_acled_{country}_state_actors.csv` | Error correlations |

### Harm Analysis
| File | Description |
|------|-------------|
| `harm_metrics_detailed.csv` | FL/FI rates |
| `fl_fi_by_model.csv` | Aggregated harm metrics |

### Error Analysis
| File | Description |
|------|-------------|
| `per_class_report.csv` | Per-class metrics |
| `top_disagreements.csv` | Model disagreements |
| `error_cases_false_legitimization.csv` | Sampled FL errors |
| `error_cases_false_illegitimization.csv` | Sampled FI errors |

### Counterfactual
| File | Description |
|------|-------------|
| `counterfactual_analysis_{models}.json` | Full analysis |
| `counterfactual_analysis_{models}_summary.csv` | Summary table |
