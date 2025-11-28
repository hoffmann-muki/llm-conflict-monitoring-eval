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

## Environment Variables

All modules use consistent environment variable configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `COUNTRY` | Country code (cmr, nga) | cmr |
| `STRATEGY` | Prompting strategy (zero_shot, few_shot, explainable) | zero_shot |
| `SAMPLE_SIZE` | Number of events sampled | 500 |
| `NUM_EXAMPLES` | Few-shot examples per category (1-5, only for few_shot) | None |

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
from lib.core.data_helpers import (
    setup_country_environment,
    paths_for_country,
    get_strategy,
    get_sample_size,
    get_num_examples,
    write_sample
)
from lib.core.metrics_helpers import aggregate_fl_fi, LEGIT, ILLEG
from lib.core.result_aggregator import aggregate_model_results, get_per_model_result_path
```

**Environment Setup (Recommended):**

```python
from lib.core.data_helpers import setup_country_environment, get_strategy, get_sample_size

# Get country and results directory from environment
country, results_dir = setup_country_environment()
# Returns: ('cmr', 'results/cmr/zero_shot/500')

# For few_shot with NUM_EXAMPLES=3:
# Returns: ('cmr', 'results/cmr/few_shot/500/3')

# Get individual values
strategy = get_strategy()       # 'zero_shot'
sample_size = get_sample_size() # '500'
num_examples = get_num_examples()  # None or 1-5
```

**Path Resolution:**

```python
from lib.core.data_helpers import paths_for_country

paths = paths_for_country('cmr')
# Returns:
# {
#   'results_dir': 'results/cmr/zero_shot/500',
#   'datasets_dir': 'datasets/cmr',
#   'sample_path': 'datasets/cmr/state_actor_sample_cmr_500.csv',
#   'calibrated_csv': 'results/cmr/zero_shot/500/ollama_results_calibrated.csv'
# }
```

**Result Aggregator:**

Combines per-model inference files into a single aggregated file for cross-model analysis:

```bash
# Aggregate all per-model files
COUNTRY=cmr STRATEGY=zero_shot SAMPLE_SIZE=500 python -m lib.core.result_aggregator
```

```python
from lib.core.result_aggregator import get_per_model_result_path, model_name_to_slug

# Get path for specific model's results
path = get_per_model_result_path('cmr', 'mistral:7b', strategy='zero_shot', sample_size='500')
# Returns: results/cmr/zero_shot/500/ollama_results_mistral-7b_acled_cmr_state_actors.csv

# Convert model name to filename-safe slug
slug = model_name_to_slug('llama3.1:8b')  # Returns: 'llama3.1-8b'
```

**Sample File Management:**

```python
from lib.core.data_helpers import write_sample

# Write sample for cross-model consistency
path = write_sample('cmr', sample_df, sample_size='500')
# Creates: datasets/cmr/state_actor_sample_cmr_500.csv
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
COUNTRY=cmr STRATEGY=zero_shot SAMPLE_SIZE=500 python -m lib.analysis.metrics

# Calibration (isotonic + temperature scaling)
COUNTRY=cmr STRATEGY=zero_shot SAMPLE_SIZE=500 python -m lib.analysis.calibration

# Harm analysis (FL/FI rates)
COUNTRY=cmr STRATEGY=zero_shot SAMPLE_SIZE=500 python -m lib.analysis.harm

# Per-class metrics and error sampling
COUNTRY=cmr STRATEGY=zero_shot SAMPLE_SIZE=500 python -m lib.analysis.per_class_metrics

# Counterfactual perturbation testing
COUNTRY=cmr STRATEGY=zero_shot SAMPLE_SIZE=500 python -m lib.analysis.counterfactual --events 20

# Specify models explicitly
COUNTRY=cmr STRATEGY=zero_shot SAMPLE_SIZE=500 python -m lib.analysis.counterfactual \
  --models llama3.2,mistral:7b --events 20

# Use percentage of available disagreements
COUNTRY=cmr STRATEGY=zero_shot SAMPLE_SIZE=500 python -m lib.analysis.counterfactual --top-percent 10

# Visualizations
COUNTRY=cmr STRATEGY=zero_shot SAMPLE_SIZE=500 python -m lib.analysis.visualize_reports

# Model size comparisons (includes metadata for traceability)
COUNTRY=cmr STRATEGY=zero_shot SAMPLE_SIZE=500 python -m lib.analysis.compare_models \
  --family gemma --sizes 2b,7b

# Decision thresholds
COUNTRY=cmr STRATEGY=zero_shot SAMPLE_SIZE=500 python -m lib.analysis.thresholds
```

**Few-Shot Examples:**

```bash
# Run analysis for few-shot with 3 examples
COUNTRY=cmr STRATEGY=few_shot SAMPLE_SIZE=500 NUM_EXAMPLES=3 \
  python -m lib.analysis.metrics
```

## Directory Structure

Results are organized by country, strategy, and sample size:

```
results/
├── cmr/
│   ├── zero_shot/
│   │   ├── 500/
│   │   │   ├── ollama_results_mistral-7b_acled_cmr_state_actors.csv
│   │   │   ├── ollama_results_acled_cmr_state_actors.csv
│   │   │   ├── ollama_results_calibrated.csv
│   │   │   ├── metrics_acled_cmr_state_actors.csv
│   │   │   └── ...
│   │   └── 1000/
│   │       └── ...
│   └── few_shot/
│       └── 500/
│           ├── 3/       # NUM_EXAMPLES=3
│           │   └── ...
│           └── 5/       # NUM_EXAMPLES=5
│               └── ...
└── nga/
    └── ...

datasets/
├── cmr/
│   ├── state_actor_sample_cmr_500.csv    # Unified sample for 500 events
│   └── state_actor_sample_cmr_1000.csv   # Unified sample for 1000 events
└── nga/
    └── ...
```

## Output Files

Each analysis module writes to `results/<country>/<strategy>/<sample_size>/`:

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

### Model Comparison
| File | Description |
|------|-------------|
| `compare_{family}_sizes.csv` | FL/FI by model with metadata |
| `compare_{family}_pairwise.csv` | McNemar test results with context |
