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

## Core Module

### Environment and Path Helpers

```python
from lib.core.data_helpers import (
    setup_country_environment,  # Returns (country, results_dir)
    paths_for_country,          # Returns dict with all standard paths
    get_strategy,               # Get STRATEGY env var
    get_sample_size,            # Get SAMPLE_SIZE env var
    get_num_examples,           # Get NUM_EXAMPLES env var
    write_sample                # Write sample file for cross-model reuse
)
```

**Usage:**

```python
# Get country and results directory from environment
country, results_dir = setup_country_environment()
# Returns: ('cmr', 'results/cmr/zero_shot/500')
# For few_shot with NUM_EXAMPLES=3: ('cmr', 'results/cmr/few_shot/500/3')

# Get all standard paths
paths = paths_for_country('cmr')
# Returns: {
#   'results_dir': 'results/cmr/zero_shot/500',
#   'datasets_dir': 'datasets/cmr',
#   'sample_path': 'datasets/cmr/state_actor_sample_cmr_500.csv',
#   'calibrated_csv': 'results/cmr/zero_shot/500/ollama_results_calibrated.csv'
# }
```

### Result Aggregation

```python
from lib.core.result_aggregator import (
    aggregate_model_results,    # Combine per-model files
    get_per_model_result_path,  # Get path for specific model
    model_name_to_slug          # Convert 'llama3.2:3b' → 'llama3.1-3b'
)
```

### Constants

```python
from lib.core.constants import LABEL_MAP, EVENT_CLASSES_FULL, WORKING_MODELS
from lib.core.metrics_helpers import aggregate_fl_fi, LEGIT, ILLEG
```

## Data Preparation

```python
from lib.data_preparation import (
    extract_country_rows,      # Extract country-specific rows
    get_actor_norm_series,     # Normalize actor names
    extract_state_actor,       # Identify state actors
    build_stratified_sample    # Create stratified samples
)
```

## Inference

```python
from lib.inference.ollama_client import run_ollama_structured
from experiments.prompting_strategies import ZeroShotStrategy

strategy = ZeroShotStrategy()
result = run_ollama_structured(
    'gemma:2b',
    strategy.make_prompt('Event description'),
    strategy.get_system_message()
)
# Returns: {"label": "V", "confidence": 0.9}
```

## Analysis Modules

All modules are runnable via `python -m`:

```bash
# Set environment
export COUNTRY=cmr STRATEGY=zero_shot SAMPLE_SIZE=500

# Run individual analyses
python -m lib.analysis.calibration      # Isotonic + temperature scaling
python -m lib.analysis.metrics          # Classification + fairness metrics
python -m lib.analysis.harm             # FL/FI rates
python -m lib.analysis.per_class_metrics
python -m lib.analysis.visualize_reports
python -m lib.analysis.thresholds
python -m lib.analysis.compare_models --family gemma --sizes 2b,7b

# Counterfactual analysis
python -m lib.analysis.counterfactual --events 20
python -m lib.analysis.counterfactual --models llama3.2,mistral:7b --top-percent 10
```

## Output Files

All output is written to `results/{country}/{strategy}/{sample_size}/`:

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
| `isotonic_mappings.json` | Calibration mappings |
| `reliability_diagrams.png` | Visualization |

### Metrics
| File | Description |
|------|-------------|
| `metrics_acled_{country}_state_actors.csv` | Classification metrics |
| `fairness_metrics_acled_{country}_state_actors.csv` | SPD, Equalized Odds |
| `confusion_matrices_acled_{country}_state_actors.json` | Confusion matrices |

### Harm Analysis
| File | Description |
|------|-------------|
| `harm_metrics_detailed.csv` | FL/FI rates by model |
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

### Model Comparison
| File | Description |
|------|-------------|
| `compare_{family}_sizes.csv` | FL/FI with metadata |
| `compare_{family}_pairwise.csv` | McNemar test results |

## Directory Structure

```
results/{country}/{strategy}/{sample_size}/
    └── {num_examples}/   # Only for few_shot strategy

datasets/{country}/
    └── state_actor_sample_{country}_{sample_size}.csv
```
