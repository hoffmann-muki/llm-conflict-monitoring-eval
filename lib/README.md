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
    get_strategy,               # Get strategy (default: 'zero_shot')
    get_sample_size,            # Get sample size (default: '500')
    get_num_examples,           # Get num_examples (default: None)
    write_sample                # Write sample file for cross-model reuse
)
```

**Usage:**

```python
# Get country and results directory with explicit arguments
country, results_dir = setup_country_environment('cmr', 'zero_shot', '500')
# Returns: ('cmr', 'results/cmr/zero_shot/500')

# For few_shot with num_examples:
country, results_dir = setup_country_environment('cmr', 'few_shot', '500', 3)
# Returns: ('cmr', 'results/cmr/few_shot/500/3')

# Get all standard paths
paths = paths_for_country('cmr', 'zero_shot', '500')
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
    build_stratified_sample,   # Create stratified samples
    build_balanced_actor_sample  # Create balanced state/non-state samples
)
```

### Balanced Actor Sampling

For fairness analysis, use balanced sampling to ensure equal representation:

```python
from lib.data_preparation.sample_builder import build_balanced_actor_sample

# Create 50/50 state vs non-state actor sample
sample = build_balanced_actor_sample(
    df,                    # Source DataFrame
    n_total=1000,          # Total sample size
    balance_ratio=0.5,     # 50% state actors
    event_types=['Violence against civilians', 'Battles'],
    min_per_cell=10        # Minimum per event_type × actor_type
)
```

## Inference

```python
from lib.inference.ollama_client import run_ollama_structured, VALID_LABELS
from experiments.prompting_strategies import ZeroShotStrategy

strategy = ZeroShotStrategy()
result = run_ollama_structured(
    'gemma:2b',
    strategy.make_prompt('Event description'),
    strategy.get_system_message()
)
# Returns: {"label": "V", "confidence": 0.9}
# Label is guaranteed to be one of VALID_LABELS: V, B, E, P, R, S
```

### Label Validation

The inference client enforces valid labels via:
1. **Ollama structured output** with JSON schema enum constraint
2. **Prompt clarity** with explicit valid label list
3. **Fallback normalization** mapping common invalid outputs to valid labels

## Analysis Modules

All modules are runnable via `python -m` with CLI arguments:

```bash
# Run individual analyses with explicit arguments
python -m lib.analysis.calibration --country cmr --strategy zero_shot --sample-size 500
python -m lib.analysis.metrics --country cmr --strategy zero_shot --sample-size 500
python -m lib.analysis.harm --country cmr --strategy zero_shot --sample-size 500
python -m lib.analysis.per_class_metrics --country cmr --strategy zero_shot --sample-size 500
python -m lib.analysis.visualize_reports --country cmr --strategy zero_shot --sample-size 500
python -m lib.analysis.thresholds --country cmr --strategy zero_shot --sample-size 500

# Model comparison
python -m lib.analysis.compare_models --country cmr --strategy zero_shot --sample-size 500 \
    --family gemma --sizes 2b,7b
python -m lib.analysis.compare_all_models --country cmr --strategy zero_shot --sample-size 500

# Counterfactual analysis
python -m lib.analysis.counterfactual --country cmr --strategy zero_shot --sample-size 500 \
    --events 20
python -m lib.analysis.counterfactual --country cmr --strategy zero_shot --sample-size 500 \
    --models llama3.2,mistral:7b --top-percent 10

# Result aggregation
python -m lib.core.result_aggregator --country cmr --strategy zero_shot --sample-size 500
```

## Output Files

All output is written to `results/{country}/{strategy}/{sample_size}/`:

### Inference
| File | Description |
|------|-------------|
| `ollama_results_{model}_acled_{country}_actors.csv` | Per-model results |
| `ollama_results_acled_{country}_actors.csv` | Combined results |
| `reasoning` column added to every CSV | Keeps explainable prompts auditable across aggregation/metrics |

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
| `metrics_acled_{country}_actors.csv` | Classification metrics |
| `fairness_metrics_acled_{country}_actors.csv` | SPD, Equalized Odds |
| `confusion_matrices_acled_{country}_actors.json` | Confusion matrices |

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
| `error_correlations_acled_{country}_actors.csv` | Text feature correlations with errors |

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

### Comparison Folder
| File/Folder | Description |
|-------------|-------------|
| `comparison/all_models_metrics.csv` | Combined accuracy/metric table for Ollama + ConfliBERT |
| `comparison/all_models_harm.csv` | Harm metrics from `lib.analysis.compare_all_models` |
| `comparison/all_models_fairness.csv` | Fairness breakdown for each model |
| `comparison/*.png` | Visualization comparing Ollama and ConfliBERT runs |

## Directory Structure

```
results/{country}/{strategy}/{sample_size}/
    └── {num_examples}/   # Only for few_shot strategy

datasets/{country}/
    └── state_actor_sample_{country}_{sample_size}.csv
```
