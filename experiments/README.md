# Experiments

`experiments/` contains the runnable workflows for our paper **Are LLMs Ready for Conflict Monitoring? Empirical Evidence from West Africa** ([arXiv:2605.04177](https://arxiv.org/abs/2605.04177)). These scripts reproduce and extend our prompt-based inference, supervised domain adaptation, split creation, SFT data preparation, held-out evaluation, and prompt-strategy experiments.

Our experimental design compares vanilla open-weight LLMs against domain-adapted baselines. In the paper, we refer to the fine-tuned ConfliBERT baseline as **AfroConfliBERT** and to the fine-tuned local causal LM baseline as **AfroConfliLLAMA**. The repository keeps some earlier engineering names, such as `conflibert_finetuned_*`, `small_llm_merged_*`, and `acled-small-llm-ft:v1`.

## Layout

```text
experiments/
|-- data/few_shot/              # Few-shot CSV bundles used by older ConfliBERT fine-tuning flows
|-- pipelines/
|   |-- conflibert/             # Download, classify, fine-tune, evaluate, and few-shot selection scripts
|   |-- data/                   # Leak-safe ACLED train/dev/test split builder
|   `-- ollama/                 # Ollama/HF classification, SFT prep, LoRA fine-tuning, registration, eval
|-- prompting_strategies/       # zero_shot, few_shot, explainable strategy classes
`-- scripts/                    # End-to-end shell orchestrators
```

## Prompt-Experiment Workflow

The main prompt workflow supports our analysis of false illegitimation, actor-based selection bias, lexical fragility, and error-trace behavior. It is per-model-then-aggregate:

1. Create or reuse `datasets/{country}/state_actor_sample_{country}_{sample_size}.csv`.
2. Run one or more models against identical event rows.
3. Save each model under `results/{country}/{strategy}/{sample_size}/{model_slug}/`.
4. Aggregate per-model files into `ollama_results_acled_{country}_actors.csv`.
5. Run calibration, metrics, fairness, harm, thresholds, per-class reports, counterfactuals, word impacts, and error traces.

Few-shot results add an extra shot-count directory:

```text
results/{country}/few_shot/{sample_size}/{num_examples}/
```

## `run_full_analysis.sh`

Canonical prompt-experiment command:

```bash
COUNTRY=cmr SAMPLE_SIZE=1000 STRATEGY=zero_shot \
  ./experiments/scripts/run_full_analysis.sh
```

Useful environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `COUNTRY` | `cmr` | Country code: `cmr` or `nga`. |
| `SAMPLE_SIZE` | `500` | Number of sampled events. |
| `STRATEGY` | `zero_shot` | `zero_shot`, `few_shot`, or `explainable`. |
| `NUM_EXAMPLES` | `3` | Few-shot examples per category when `STRATEGY=few_shot`. |
| `INFERENCE_MODELS` | all `WORKING_MODELS` | Comma-separated model names to run. |
| `ANALYSIS_MODELS` | `all` | Optional comma-separated subset for aggregation/analysis. |
| `HF_INFERENCE_MODELS` | empty | Comma-separated model names that should run from local HF checkpoints. |
| `HF_MODEL_PATH` | empty | Single HF checkpoint path fallback. |
| `HF_MODEL_PATH_MAP` | empty | Comma-separated `model=/path` map for multiple HF-backed names. |
| `HF_DEVICE` | auto | HF device override, for example `cpu`, `cuda`, or `cuda:0`. |
| `HF_MAX_NEW_TOKENS` | `96` | Generation budget for HF-backed inference. |
| `CF_MODELS` | `INFERENCE_MODELS` when set, else all working models | Models for counterfactual analysis. |
| `CF_EVENTS` | `20` | Number of high-disagreement events used for counterfactuals. |
| `SKIP_INFERENCE` | `false` | Reuse existing per-model files. |
| `SKIP_COUNTERFACTUAL` | `false` | Skip counterfactual, word-impact, and trace phases. |

Examples:

```bash
# Few-shot prompting.
COUNTRY=nga SAMPLE_SIZE=1000 STRATEGY=few_shot NUM_EXAMPLES=3 \
  ./experiments/scripts/run_full_analysis.sh

# Specific Ollama models.
COUNTRY=cmr SAMPLE_SIZE=1000 INFERENCE_MODELS="mistral:7b,llama3.2:3b" \
  ./experiments/scripts/run_full_analysis.sh

# HF-backed fine-tuned model plus an Ollama model.
COUNTRY=nga SAMPLE_SIZE=1000 STRATEGY=zero_shot \
INFERENCE_MODELS="acled-small-llm-ft:v1,mistral:7b" \
HF_INFERENCE_MODELS="acled-small-llm-ft:v1" \
HF_MODEL_PATH_MAP="acled-small-llm-ft:v1=models/small_llm_merged_acled_v1_seed42" \
  ./experiments/scripts/run_full_analysis.sh

# Analyze existing per-model outputs only.
COUNTRY=cmr SAMPLE_SIZE=1000 SKIP_INFERENCE=true \
  ./experiments/scripts/run_full_analysis.sh
```

`run_ollama_full_analysis.sh` is only a compatibility wrapper around `run_full_analysis.sh`.

## `run_calibrate_then_apply.sh`

This helper performs a two-stage threshold workflow:

1. Run a small sample, aggregate results, calibrate, and select thresholds.
2. Run a larger sample, calibrate, apply the selected thresholds, and generate reports.

```bash
COUNTRY=cmr STRATEGY=zero_shot SMALL_SAMPLE=20 LARGE_SAMPLE=50 MIN_COVERAGE=0.5 \
  ./experiments/scripts/run_calibrate_then_apply.sh
```

It writes the large-sample report to:

```text
results/{country}/{strategy}/{large_sample}/final_threshold_performance.csv
```

## Direct Prompt Inference

Ollama/HF pipeline:

```bash
python experiments/pipelines/ollama/run_ollama_classification.py cmr \
  --sample-size 1000 \
  --strategy zero_shot \
  --models "mistral:7b,llama3.2:3b"
```

Few-shot:

```bash
python experiments/pipelines/ollama/run_ollama_classification.py nga \
  --sample-size 1000 \
  --strategy few_shot \
  --num-examples 3 \
  --models "gemma3:4b"
```

Targeted sampling can prioritize one event type while preserving actor balance:

```bash
python experiments/pipelines/ollama/run_ollama_classification.py cmr \
  --sample-size 1000 \
  --primary-group "Violence against civilians" \
  --primary-share 0.6
```

Standalone ConfliBERT classification:

```bash
python experiments/pipelines/conflibert/run_conflibert_classification.py cmr \
  --sample-size 1000 \
  --model-path models/conflibert \
  --model-name conflibert
```

The ConfliBERT classification script currently forces `strategy=zero_shot` because ConfliBERT is treated as a supervised classifier rather than a prompt-conditioned chat model.

## Fine-Tuned Baselines

The supervised baseline runner builds our leak-safe train/dev/test split bundles, fine-tunes/evaluates ConfliBERT, optionally fine-tunes/evaluates a small causal LLM with LoRA SFT, then builds comparison tables. These workflows correspond to the domain-adapted models we report as AfroConfliBERT and AfroConfliLLAMA.

```bash
SPLIT_VERSION=acled_v1 ./experiments/scripts/run_finetuned_baselines.sh
```

Small-LLM example:

```bash
SPLIT_VERSION=acled_v1 RUN_SMALL_LLM=true \
SMALL_LLM_BASE_MODEL=models/Llama-3.2-3B \
SMALL_LLM_STRATEGY=zero_shot \
  ./experiments/scripts/run_finetuned_baselines.sh
```

Key baseline variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SPLIT_VERSION` | required | Split identifier under `data/processed/splits/`. |
| `EVAL_COUNTRIES` | `cmr,nga` | Held-out country splits to create/evaluate. |
| `RUN_CONFLIBERT` | `true` | Fine-tune and evaluate ConfliBERT. |
| `RUN_SMALL_LLM` | `false` | Enable LoRA SFT for a local causal LM. |
| `BALANCE_TRAIN` | `true` | Class-balance train split. |
| `BALANCE_DEV` | `false` | Class-balance dev split. |
| `BALANCE_TEST` | `false` | Class-balance held-out test splits. |
| `TRAIN_MAX_PER_CLASS` | `0` | Optional train cap per class. |
| `TEST_MAX_PER_COUNTRY` | `0` | Optional held-out cap per country. |
| `CONFLIBERT_BASE_MODEL` | `models/conflibert` | Local base ConfliBERT directory. |
| `CONFLIBERT_EPOCHS` | `4` | ConfliBERT fine-tuning epochs. |
| `SMALL_LLM_BASE_MODEL` | required when small LLM is enabled | Local HF causal LM directory. |
| `SMALL_LLM_SKIP_FINETUNE` | `false` | Reuse an existing merged model. |
| `SMALL_LLM_MERGED_MODEL_DIR` | empty | Existing merged model path when skipping. |
| `SMALL_LLM_EVAL_MAX_SAMPLES` | `0` | Optional eval cap per held-out split. |

Baseline outputs:

```text
data/processed/splits/{split_version}/
results/baselines/{split_version}/
models/conflibert_finetuned_{split_version}_seed{seed}/
models/small_llm_adapter_{split_version}_seed{seed}/
models/small_llm_merged_{split_version}_seed{seed}/
```

## Split Builder

Build leak-safe train/dev/test splits directly:

```bash
python experiments/pipelines/data/build_train_dev_test_splits.py \
  --split-version acled_v1 \
  --eval-countries cmr,nga \
  --dev-ratio 0.1 \
  --balance-train \
  --train-max-per-class 8000 \
  --test-max-per-country 3000
```

Outputs:

```text
data/processed/splits/{split_version}/train.csv
data/processed/splits/{split_version}/dev.csv
data/processed/splits/{split_version}/test_cmr.csv
data/processed/splits/{split_version}/test_nga.csv
data/processed/splits/{split_version}/manifest.json
```

## Small-LLM SFT Pieces

Prepare JSONL from split CSVs:

```bash
python experiments/pipelines/ollama/prepare_sft_data.py \
  --input-csv data/processed/splits/acled_v1/train.csv \
  --output-jsonl results/baselines/acled_v1/sft_data/train.jsonl \
  --strategy zero_shot
```

Fine-tune with LoRA:

```bash
python experiments/pipelines/ollama/finetune_small_llm.py \
  --train-jsonl results/baselines/acled_v1/sft_data/train.jsonl \
  --dev-jsonl results/baselines/acled_v1/sft_data/dev.jsonl \
  --base-model models/Llama-3.2-3B \
  --output-dir models/small_llm_adapter_acled_v1_seed42 \
  --merged-output-dir models/small_llm_merged_acled_v1_seed42
```

Evaluate a merged HF model on a fixed split:

```bash
python experiments/pipelines/ollama/evaluate_ollama_on_split.py \
  --model-path models/small_llm_merged_acled_v1_seed42 \
  --model-name small_llm_merged_acled_v1_seed42 \
  --input-csv data/processed/splits/acled_v1/test_nga.csv \
  --output-csv results/baselines/acled_v1/small_llm/hf_predictions_nga.csv
```

Optional Ollama registration helper:

```bash
python experiments/pipelines/ollama/register_ollama_model.py \
  --merged-model-dir models/small_llm_merged_acled_v1_seed42 \
  --ollama-model-name acled-small-llm-ft:v1 \
  --dry-run
```

## Prompting Strategies

Available strategies:

| Strategy | Description | Extra config |
| --- | --- | --- |
| `zero_shot` | Direct classification with label descriptions and JSON schema. | none |
| `few_shot` | Demonstrations for each label before classification. | `NUM_EXAMPLES` / `--num-examples`, 1 to 5 |
| `explainable` | Requires three short reasoning items plus label/confidence/logits. | none |

See `experiments/prompting_strategies/README.md` for the class interface and registration steps.
