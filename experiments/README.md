# Experiments

Pipelines, prompting strategies, and shell scripts for running classification experiments. Scripts reuse the shared samples plus the JSON schema so that Ollama and ConfliBERT always see the same rows and structured outputs.

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

The pipeline follows a **per-model-then-aggregate** design so the various models share the same sample, aggregation, and analysis flow:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Sample Creation                                                             │
│   Unified sample: datasets/{country}/state_actor_sample_{country}_{n}.csv   │
│   Same sample reused across all models (random_state=42)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Per-Model Inference → Aggregation → Analysis (Calibration, Metrics, Harm)   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Benefits:** Fair comparison, reproducibility, and consistent per-strategy isolation.

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

# Hybrid inference: run fine-tuned model from local HF checkpoint (no Ollama registration)
OLLAMA_MODELS=acled-small-llm-ft:v1 COUNTRY=nga SAMPLE_SIZE=1000 STRATEGY=zero_shot \
HF_INFERENCE_MODELS=acled-small-llm-ft:v1 \
HF_MODEL_PATH=/absolute/path/to/models/small_llm_merged_acled_v1_seed42 \
  ./experiments/scripts/run_ollama_full_analysis.sh

# Hybrid multi-model inference: one HF model + regular Ollama models
OLLAMA_MODELS=acled-small-llm-ft:v1,mistral:7b,llama3.2:3b COUNTRY=cmr SAMPLE_SIZE=500 \
HF_INFERENCE_MODELS=acled-small-llm-ft:v1 \
HF_MODEL_PATH_MAP='acled-small-llm-ft:v1=/absolute/path/to/models/small_llm_merged_acled_v1_seed42' \
  ./experiments/scripts/run_ollama_full_analysis.sh

# Note: when CF_MODELS is unset, counterfactual and error-trace now inherit
# OLLAMA_MODELS, so HF-backed fine-tuned models remain included end-to-end.

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

Two-stage calibration: calibrate on a small sample and apply the mapping to the larger inference dataset.

```bash
COUNTRY=cmr STRATEGY=zero_shot SMALL_SAMPLE=20 LARGE_SAMPLE=50 \
  ./experiments/scripts/run_calibrate_then_apply.sh
```

### run_conflibert_experiment.sh

ConfliBERT experiment with the same interface as the Ollama scripts.

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

# HF-backed model without Ollama API for that model name
HF_INFERENCE_MODELS=acled-small-llm-ft:v1 \
HF_MODEL_PATH=/absolute/path/to/models/small_llm_merged_acled_v1_seed42 \
python experiments/pipelines/ollama/run_ollama_classification.py nga \
  --sample-size 1000 --models "acled-small-llm-ft:v1"
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

For fair comparison, sample files are created once and reused:

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

Results are written to `results/{country}/{strategy}/{sample_size}/`. For few-shot, an extra `{num_examples}/` folder is created.

Every per-model results CSV now includes a `reasoning` column, so explainable prompts remain auditable as they flow through aggregation, calibration, and metrics. After running both Ollama and ConfliBERT experiments, execute `python -m lib.analysis.compare_all_models` to fill `results/.../comparison/` with the combined metrics, fairness, harm tables, and comparison PNGs.

See [lib/README.md](../lib/README.md) for detailed output documentation.
