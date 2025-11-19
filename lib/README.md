# Library

Reusable library components for LLM state actor bias analysis.

## Structure

```
lib/
├── data_preparation/  # Data extraction, normalization, sampling
├── inference/         # Ollama model inference
├── analysis/          # Metrics, calibration, fairness, counterfactual
└── core/              # Constants, helpers, utilities
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

**Functions:**
- `extract_country_rows()` - Extract country-specific rows from ACLED data
- `get_actor_norm_series()` - Normalize actor names
- `extract_state_actor()` - Identify state actors
- `build_stratified_sample()` - Create stratified samples with oversampling

### Inference

Ollama model inference with structured JSON output:

```python
from lib.inference.ollama_client import run_ollama_structured

result = run_ollama_structured('gemma:2b', 'Event description')
# Returns: {"label": "V", "confidence": 0.9}
```

### Analysis

All analysis modules are runnable:

```bash
# Classification metrics and fairness
python -m lib.analysis.metrics

# Calibration (isotonic + temperature scaling)
python -m lib.analysis.calibration

# Harm analysis (FL/FI rates)
python -m lib.analysis.harm

# Per-class metrics and error sampling
python -m lib.analysis.per_class

# Counterfactual perturbation testing
python -m lib.analysis.counterfactual --models llama3.2,mistral:7b --events 50

# Visualizations
python -m lib.analysis.visualize_reports
python -m lib.analysis.visualize_counterfactual

# Model size comparisons
python -m lib.analysis.compare_models --family gemma --sizes 2b,7b

# Decision thresholds
python -m lib.analysis.thresholds
```

### Core

Constants and shared utilities:

```python
from lib.core.constants import LABEL_MAP, EVENT_CLASSES_FULL, WORKING_MODELS
from lib.core.data_helpers import setup_country_environment, paths_for_country
from lib.core.metrics_helpers import aggregate_fl_fi, LEGIT, ILLEG
```

## Environment Variables

Most analysis modules use:
- `COUNTRY` - Country code (cmr, nga)
- `RESULTS_DIR` - Results directory path (optional, auto-detected from COUNTRY)

## Output Files

Each analysis module writes to `results/<COUNTRY>/<STRATEGY>/`:

**Calibration:**
- `ollama_results_calibrated.csv`
- `calibration_brier_scores.csv`
- `reliability_diagrams.png`

**Metrics:**
- `metrics_acled_<country>_state_actors.csv`
- `fairness_metrics_acled_<country>_state_actors.csv`
- `confusion_matrices_acled_<country>_state_actors.json`

**Harm:**
- `harm_metrics_detailed.csv`
- `fl_fi_by_model.csv`

**Error Analysis:**
- `error_cases_false_legitimization.csv`
- `error_cases_false_illegitimization.csv`
- `error_correlations_acled_<country>_state_actors.csv`

**Counterfactual:**
- `counterfactual_analysis_<models>.json`
- `counterfactual_analysis_<models>_summary.csv`
