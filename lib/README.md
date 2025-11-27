# Library

Reusable library components for LLM state actor bias analysis.

## Structure

```
lib/
├── data_preparation/  # Data extraction, normalization, sampling
├── inference/         # Ollama model inference
├── conflibert/        # ConfliBERT classification (HuggingFace models)
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

#### Ollama Models

Ollama model inference with structured JSON output.

**Important:** `run_ollama_structured()` requires an explicit `prompt` parameter. Always use prompting strategies to generate prompts - there are no hardcoded prompts.

```python
from lib.inference.ollama_client import run_ollama_structured
from experiments.prompting_strategies import ZeroShotStrategy

strategy = ZeroShotStrategy()
prompt = strategy.make_prompt('Event description')
system_msg = strategy.get_system_message()
result = run_ollama_structured('gemma:2b', prompt, system_msg)
# Returns: {"label": "V", "confidence": 0.9}
```

#### ConfliBERT

ConfliBERT classification integrated with the same prompting strategy framework:

```bash
# Download model first (one-time setup)
python experiments/pipelines/conflibert/download_conflibert_model.py --out-dir models/conflibert

# Run ConfliBERT classification
python experiments/pipelines/conflibert/run_conflibert_classification.py cmr \
  --model-path models/conflibert --strategy zero_shot --sample-size 100

# Compare with Ollama models
python -m lib.analysis.per_class_metrics cmr zero_shot
```

**Key Points:**
- Requires local model path (use `download_conflibert_model.py` to fetch)
- Uses same strategy interface (zero_shot, few_shot, explainable) for organization
- Outputs results in identical format to Ollama pipeline
- Works with all downstream analysis tools (per_class_metrics, counterfactual, etc.)

### Analysis

All analysis modules are runnable as Python modules:

```bash
# Classification metrics and fairness
COUNTRY=cmr python -m lib.analysis.metrics

# Calibration (isotonic + temperature scaling)
COUNTRY=cmr python -m lib.analysis.calibration

# Harm analysis (FL/FI rates)
COUNTRY=cmr python -m lib.analysis.harm

# Per-class metrics and error sampling
COUNTRY=cmr python -m lib.analysis.per_class_metrics

# Counterfactual perturbation testing (requires top_disagreements.csv)
# Generate disagreements first and then run counterfactual on the top-N or top-percent:
COUNTRY=cmr python -m lib.analysis.per_class_metrics  # Generate disagreements first
COUNTRY=cmr python -m lib.analysis.counterfactual --events 20  # Uses all WORKING_MODELS

# Or specify models explicitly:
COUNTRY=cmr python -m lib.analysis.counterfactual \
  --models llama3.2,mistral:7b --events 20

# Or use a percentage of available disagreements (e.g., top 10%):
COUNTRY=cmr python -m lib.analysis.counterfactual --top-percent 10

# Visualizations
COUNTRY=cmr python -m lib.analysis.visualize_reports
COUNTRY=cmr python -m lib.analysis.visualize_counterfactual \
  --input results/cmr/zero_shot/counterfactual_analysis_*.json

# Model size comparisons
COUNTRY=cmr python -m lib.analysis.compare_models \
  --family gemma --sizes 2b,7b

# Decision thresholds
COUNTRY=cmr python -m lib.analysis.thresholds
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
