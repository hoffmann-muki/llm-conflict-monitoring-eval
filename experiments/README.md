# Reproducing the experiments

This guide covers three operations: generating model predictions, training domain-adapted baselines, and rebuilding reports. Complete the [installation](../README.md#quick-start) first, activate the environment, and issue all commands from the repository root.

## Evaluation protocols

**Prompt comparison.** Each country has a cached event sample shared across models and prompting conditions. New samples target equal state/non-state representation, stratify by event type, and use seed 42. A sample is identified by country and requested size at `datasets/{country}/state_actor_sample_{country}_{size}.csv`. Once that file exists, the runner reuses it, including when sampling options change.

**Supervised evaluation.** The split builder deduplicates event IDs, reserves Cameroon and Nigeria for testing by default, and partitions the remaining training pool into train/dev sets. It checks that event IDs do not overlap across these sets. The supplied [acled_v1 manifest](../data/processed/splits/acled_v1/manifest.json) records the seed, balancing choices, sizes, and label distributions. These test sets are distinct from the cached prompt samples.

## Generate predictions and diagnostics

Start an Ollama service and obtain the model before launching a prompt experiment:

```bash
ollama pull mistral:7b

COUNTRY=cmr SAMPLE_SIZE=1000 STRATEGY=zero_shot \
INFERENCE_MODELS="mistral:7b" ANALYSIS_MODELS="mistral:7b" \
  bash experiments/scripts/run_full_analysis.sh
```

The runner writes predictions for each model, combines them, calibrates confidence, and computes performance, actor fairness, and directional harm. It then perturbs high-disagreement events and produces robustness and error-trace analyses. Some post-processing stages are non-fatal; their completion status appears in the terminal output.

Choose a condition through the following controls:

| Control | Values and behavior |
| --- | --- |
| `COUNTRY` | `cmr` for Cameroon; `nga` for Nigeria. Default: `cmr`. |
| `SAMPLE_SIZE` | Requested event count. Default: `500`; supplied prompt samples use `1000`. |
| `STRATEGY` | `zero_shot`, `few_shot`, or `explainable`. Default: `zero_shot`. |
| `NUM_EXAMPLES` | Few-shot demonstrations **per class**, from 1 to 5. Shell default: `3`. |
| `INFERENCE_MODELS` | Comma-separated model IDs for prediction. Set explicitly; the default model list includes a separately provisioned adapted model. |
| `ANALYSIS_MODELS` | Comma-separated model IDs to aggregate. Default: all model files already in the condition directory. |
| `CF_MODELS` / `CF_EVENTS` | Models and event budget for perturbations. Defaults: the inference selection, or the working model list if unset; 20 events. |

For example, append `STRATEGY=few_shot NUM_EXAMPLES=3` to the environment assignments above to evaluate three demonstrations per class. The [prompt definitions](prompting_strategies/README.md) specify exactly what changes between conditions.

For a merged Hugging Face checkpoint, use the same runner with an explicit backend mapping:

```bash
COUNTRY=nga SAMPLE_SIZE=1000 STRATEGY=zero_shot \
INFERENCE_MODELS="acled-small-llm-ft:v1" \
ANALYSIS_MODELS="acled-small-llm-ft:v1" \
HF_INFERENCE_MODELS="acled-small-llm-ft:v1" \
HF_MODEL_PATH="models/small_llm_merged_acled_v1_seed42" \
  bash experiments/scripts/run_full_analysis.sh
```

For several HF models, replace the fallback path with `HF_MODEL_PATH_MAP="model_a=path_a,model_b=path_b"`. `HF_DEVICE` selects a device and `HF_MAX_NEW_TOKENS` sets the generation budget (default: 96). The orchestrators enable offline HF loading, so checkpoints must already be local.

## Train the adapted baselines

AfroConfliBERT uses the sequence-classification workflow under [pipelines/conflibert/](pipelines/conflibert/). Obtain its base checkpoint, then launch training and country-level evaluation:

```bash
python experiments/pipelines/conflibert/download_conflibert_model.py \
  --out-dir models/conflibert

SPLIT_VERSION=acled_v1 \
  bash experiments/scripts/run_finetuned_baselines.sh
```

AfroConfliLLAMA uses the SFT preparation, LoRA training, checkpoint merging, and evaluation tools under [pipelines/ollama/](pipelines/ollama/). To include it in the baseline run, provide the base model locally:

```bash
SPLIT_VERSION=acled_v1 RUN_SMALL_LLM=true \
SMALL_LLM_BASE_MODEL=models/Llama-3.2-3B \
  bash experiments/scripts/run_finetuned_baselines.sh
```

The baseline runner rebuilds the selected split bundle. Give `SPLIT_VERSION` a new name when retaining a previous experiment. Default settings use seed 42, a 10% dev split, balanced training classes, and unbalanced dev/test classes. ConfliBERT trains for four epochs; causal-LM SFT trains for three. All training controls are listed in [run_finetuned_baselines.sh](scripts/run_finetuned_baselines.sh).

To evaluate an existing causal-LM checkpoint without retraining, set `RUN_SMALL_LLM=true SMALL_LLM_SKIP_FINETUNE=true` and supply `SMALL_LLM_MERGED_MODEL_DIR`. To run stages separately against existing splits, use the scripts below with `--help`:

| Operation | Entry point |
| --- | --- |
| Construct supervised splits | [build_train_dev_test_splits.py](pipelines/data/build_train_dev_test_splits.py) |
| Fit ConfliBERT / evaluate a split | [finetune_conflibert.py](pipelines/conflibert/finetune_conflibert.py) / [evaluate_conflibert_on_split.py](pipelines/conflibert/evaluate_conflibert_on_split.py) |
| Prepare SFT data / fit LoRA | [prepare_sft_data.py](pipelines/ollama/prepare_sft_data.py) / [finetune_small_llm.py](pipelines/ollama/finetune_small_llm.py) |
| Evaluate a merged causal LM | [evaluate_ollama_on_split.py](pipelines/ollama/evaluate_ollama_on_split.py) |

## Locate and rebuild outputs

A prompt condition is stored under `results/{country}/{strategy}/{size}/`; few-shot conditions add `/{num_examples}/`. Within it, each model has a subdirectory, followed by shared analysis files. For example:

```text
results/cmr/zero_shot/1000/
├── mistral_7b/ollama_results_mistral-7b_acled_cmr_actors.csv
├── ollama_results_acled_cmr_actors.csv
├── ollama_results_calibrated.csv
└── ... tables, figures, and counterfactual reports
```

Reuse the saved per-model CSVs by setting `SKIP_INFERENCE=true`. Also set `SKIP_COUNTERFACTUAL=true` to rebuild reports without invoking models. These options leave the existing prediction files as inputs and update derived artifacts; `ANALYSIS_MODELS` still determines which models enter aggregation.

Supervised predictions and comparison tables are written to `results/baselines/{split_version}/`. Checkpoints use `models/conflibert_finetuned_{split_version}_seed{seed}/` and `models/small_llm_{adapter,merged}_{split_version}_seed{seed}/`.

See the [analysis reference](../lib/README.md) for report dependencies and evaluation subsets. The separate [threshold-transfer helper](scripts/run_calibrate_then_apply.sh) fits thresholds on a smaller sample and applies them to a larger one; those independently sampled event sets may overlap.
