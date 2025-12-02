# Experiments

Pipelines, prompting strategies, and shell scripts for running classification experiments.

## Structure

```
experiments/
├── pipelines/
│   ├── ollama/              # Ollama LLM classification
│   └── conflibert/          # ConfliBERT transformer classification
├── prompting_strategies/    # Modular prompting strategies
└── scripts/                 # Shell scripts for experiments
```

## Workflow Architecture

The pipeline uses a **per-model-then-aggregate** workflow:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Sample Creation                                                             │
│   Unified sample: datasets/{country}/state_actor_sample_{country}_{n}.csv   │
│   Same sample reused across all models (random_state=42)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Per-Model Inference → Aggregation → Analysis (Calibration, Metrics, Harm)   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Benefits:** Fair comparison across models, incremental runs, reproducibility, strategy isolation.

## Shell Scripts

### run_ollama_full_analysis.sh

Complete 5-phase pipeline: Inference → Aggregation → Calibration → Metrics → Counterfactual

```bash
# Full run with all models
COUNTRY=cmr SAMPLE_SIZE=500 STRATEGY=zero_shot \
  ./experiments/scripts/run_ollama_full_analysis.sh

# Few-shot with 3 examples per category
COUNTRY=cmr SAMPLE_SIZE=500 STRATEGY=few_shot NUM_EXAMPLES=3 \
  ./experiments/scripts/run_ollama_full_analysis.sh

# Specific models only
OLLAMA_MODELS=mistral:7b,llama3.2:3b COUNTRY=nga SAMPLE_SIZE=1000 \
  ./experiments/scripts/run_ollama_full_analysis.sh

# Skip inference, analyze existing results
SKIP_INFERENCE=true COUNTRY=cmr STRATEGY=zero_shot SAMPLE_SIZE=500 \
  ./experiments/scripts/run_ollama_full_analysis.sh

# Custom counterfactual settings
CF_MODELS=mistral:7b CF_EVENTS=100 COUNTRY=cmr SAMPLE_SIZE=500 \
  ./experiments/scripts/run_ollama_full_analysis.sh

# Skip counterfactual analysis entirely
SKIP_COUNTERFACTUAL=true COUNTRY=cmr SAMPLE_SIZE=500 \
  ./experiments/scripts/run_ollama_full_analysis.sh
```

### run_calibrate_then_apply.sh

Two-stage calibration: calibrate on small sample, apply to larger sample.

```bash
COUNTRY=cmr STRATEGY=zero_shot SMALL_SAMPLE=20 LARGE_SAMPLE=50 \
  ./experiments/scripts/run_calibrate_then_apply.sh
```

### run_conflibert_experiment.sh

ConfliBERT experiment with same interface as Ollama scripts.

```bash
COUNTRY=cmr SAMPLE_SIZE=500 STRATEGY=zero_shot \
  ./experiments/scripts/run_conflibert_experiment.sh
```

## Pipelines (Python Direct)

### Ollama Pipeline

```bash
# Basic usage
python experiments/pipelines/ollama/run_ollama_classification.py cmr \
  --sample-size 500 --strategy zero_shot

# Few-shot with 3 examples
python experiments/pipelines/ollama/run_ollama_classification.py cmr \
  --sample-size 500 --strategy few_shot --num-examples 3

# Specific models
python experiments/pipelines/ollama/run_ollama_classification.py nga \
  --sample-size 1000 --models "mistral:7b,llama3.2:3b"
```

### ConfliBERT Pipeline

```bash
# Download model (one-time, ~437 MB)
python experiments/pipelines/conflibert/download_conflibert_model.py \
  --out-dir models/conflibert

# Run classification
python experiments/pipelines/conflibert/run_conflibert_classification.py cmr \
  --sample-size 500 --strategy zero_shot --model-path models/conflibert
```

## Prompting Strategies

| Strategy | Description | Config |
|----------|-------------|--------|
| `zero_shot` | Direct classification without examples | Default |
| `few_shot` | Classification with examples | `NUM_EXAMPLES=1..5` |
| `explainable` | Chain-of-thought reasoning | - |

See [prompting_strategies/README.md](prompting_strategies/README.md) for creating custom strategies.

## Sample Reuse

For fair cross-model comparison, sample files are created once and reused:

```
datasets/{country}/state_actor_sample_{country}_{sample_size}.csv
```

All models run with the same country/sample_size classify **identical events**.

## Sampling Options

**Proportional (default):** Reflects natural class distribution.

**Targeted oversampling:** Focus on specific event types:
```bash
python experiments/pipelines/ollama/run_ollama_classification.py \
  --primary-group "Violence against civilians" --primary-share 0.6
```

**Balanced actor sampling:** For fairness analysis with equal state/non-state representation (see [lib/README.md](../lib/README.md)).

## Output

Results are written to `results/{country}/{strategy}/{sample_size}/`. For few-shot, an additional subdirectory `{num_examples}/` is created.

See [lib/README.md](../lib/README.md) for complete output file documentation.
