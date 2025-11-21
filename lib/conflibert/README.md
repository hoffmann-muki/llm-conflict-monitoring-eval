# ConfliBERT Module

ConfliBERT classification integrated with the repository's prompting strategy framework.

## Overview

This module provides ConfliBERT-based classification that:
- Integrates with the repository's prompting strategy interface (for experiment organization)
- Produces results in the standard repository format
- Is compatible with downstream analysis tools (per_class_metrics, counterfactual)
- Organizes results in strategy-specific subdirectories

## Usage

### Basic Classification

Run ConfliBERT on a stratified sample with a specific strategy:

```bash
# Run ConfliBERT with zero-shot strategy
python -m lib.conflibert.classify --country cmr --strategy zero_shot --sample-size 100

# Run with few-shot strategy
python -m lib.conflibert.classify --country nga --strategy few_shot --sample-size 200

# Run with explainable strategy
python -m lib.conflibert.classify --country cmr --strategy explainable --sample-size 150
```

### Full Pipeline Integration

To run a complete comparison between Ollama models and ConfliBERT:

```bash
# 1. First, run Ollama classification to create the stratified sample
COUNTRY=cmr STRATEGY=zero_shot SAMPLE_SIZE=100 \
  python -m experiments.pipelines.run_classification

# 2. Then run ConfliBERT on the same sample
python -m lib.conflibert.classify --country cmr --strategy zero_shot --sample-size 100

# 3. Analyze results together
python -m lib.analysis.per_class_metrics cmr zero_shot
python -m lib.analysis.counterfactual --country cmr --strategy zero_shot --models llama3.1:8b,conflibert_ConfliBERT-scr-uncased --top-percent 10
```

## Command-Line Arguments

- `country` - Country code (e.g., `cmr`, `nga`)
- `--strategy` - Prompting strategy: `zero_shot`, `few_shot`, `explainable` (default: `zero_shot`)
- `--sample-size` - Expected sample size (default: 100)
- `--model` - HuggingFace model ID (default: `snowood1/ConfliBERT-scr-uncased`)
- `--batch-size` - Batch size for inference (default: 16)
- `--max-length` - Maximum sequence length (default: 256)
- `--device` - Device for inference (default: `cuda` if available, else `cpu`)

## Output Format

ConfliBERT outputs results in the same format as the Ollama pipeline:

```csv
model,event_id,true_label,pred_label,pred_conf,logits,latency_sec,actor_norm
conflibert_ConfliBERT-scr-uncased,CMR1234,V,V,0.89,"[0.89, 0.05, 0.02, 0.01, 0.02, 0.01]",0.012,Cameroon: Military Forces
```

**Columns:**
- `model` - Model identifier (e.g., `conflibert_ConfliBERT-scr-uncased`)
- `event_id` - Event identifier from ACLED data
- `true_label` - Ground truth label code (V/B/E/P/R/S)
- `pred_label` - Predicted label code
- `pred_conf` - Prediction confidence (0-1)
- `logits` - JSON array of class probabilities
- `latency_sec` - Inference time per event
- `actor_norm` - Normalized actor name

## File Organization

Results are organized by strategy to enable direct comparison with Ollama models:

```
results/
└── <country>/
    └── <strategy>/
        ├── ollama_results_acled_<country>_state_actors.csv
        └── conflibert_results_acled_<country>_state_actors.csv
```

## Requirements

Install dependencies in your virtual environment:

```bash
# Core requirements
.venv/bin/python -m pip install transformers tqdm

# PyTorch (see https://pytorch.org for platform-specific installation)
# CPU-only example:
.venv/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Integration with Analysis Tools

### Per-Class Metrics

ConfliBERT results work with the same per-class metrics tools:

```bash
# Generate metrics for both Ollama and ConfliBERT
python -m lib.analysis.per_class_metrics cmr zero_shot
```

This will produce:
- `per_class_metrics_cmr_zero_shot.csv` - Metrics for all models including ConfliBERT
- `top_disagreements.csv` - Top disagreements across all models

### Counterfactual Analysis

Run counterfactual analysis across models (including ConfliBERT) to investigate top disagreements:

```bash
# Analyze top 10% disagreements across models
python -m lib.analysis.counterfactual \
  --country cmr \
  --strategy zero_shot \
  --models llama3.1:8b,qwen3:8b,conflibert_ConfliBERT-scr-uncased \
  --top-percent 10
```

## Model Details

### Default Model

`snowood1/ConfliBERT-scr-uncased` - A BERT-based model fine-tuned for conflict event classification.

### Custom Models

You can use any HuggingFace sequence classification model:

```bash
python -m lib.conflibert.classify \
  --country cmr \
  --strategy zero_shot \
  --model your-org/your-conflict-model
```

**Requirements:**
- Model must be a sequence classification model with 6 output labels
- Labels must map to: V, B, E, P, R, S (in alphabetical order by code)

## Notes

- Use the `--strategy` argument to organize results and correlate them with other model outputs.
- Ensure the model you supply exposes six output labels corresponding to the repository taxonomy: V, B, E, P, R, S.

## Troubleshooting

### Missing Sample File

```
Error: Sample file not found: datasets/cmr/sample_cmr_state_actors.csv
```

**Solution:** Run the Ollama pipeline first to create the stratified sample:
```bash
python -m experiments.pipelines.run_classification cmr --sample-size 100 --strategy zero_shot
```

### Model Download Issues

If HuggingFace model download fails:
```bash
# Set cache directory (optional)
export HF_HOME=/path/to/cache

# Verify network access
curl -I https://huggingface.co

# Test model access
.venv/bin/python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('snowood1/ConfliBERT-scr-uncased')"
```

### CUDA/Device Issues

```bash
# Force CPU inference
python -m lib.conflibert.classify --country cmr --strategy zero_shot --device cpu

# Check CUDA availability
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"
```

## Development

### Adding New Models

To add a new ConfliBERT-like model:

1. Ensure it outputs 6 classes in the correct order
2. Update the default model if needed:
   ```python
   parser.add_argument('--model', default='your-org/your-model')
   ```
3. Verify label mapping matches your model's config

### Testing

Run on a small sample to verify integration:

```bash
# Create a small test sample
python -m experiments.pipelines.run_classification cmr --sample-size 10 --strategy zero_shot

# Run ConfliBERT
python -m lib.conflibert.classify --country cmr --strategy zero_shot --sample-size 10

# Verify output format
head -n 5 results/cmr/zero_shot/conflibert_results_acled_cmr_state_actors.csv
```
