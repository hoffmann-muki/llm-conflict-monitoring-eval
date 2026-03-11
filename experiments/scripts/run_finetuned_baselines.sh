#!/bin/bash
###############################################################################
# Fine-Tuned Baseline Runner (ConfliBERT + Small LLM)
#
# End-to-end pipeline for supervised baselines:
#   1) Build leak-safe train/dev/test splits
#   2) Fine-tune ConfliBERT on ACLED train split
#   3) Evaluate ConfliBERT on Cameroon/Nigeria held-out test splits
#   4) (Optional) Fine-tune small LLM via LoRA SFT (HF model), merge, register with Ollama, and evaluate
#
# Usage:
#   SPLIT_VERSION=acled_v1 ./experiments/scripts/run_finetuned_baselines.sh
#
# Key environment variables:
#   SPLIT_VERSION               - Required split identifier (e.g., acled_v1)
#   EVAL_COUNTRIES              - Comma list, default: cmr,nga
#   RUN_CONFLIBERT              - true/false, default: true
#   RUN_SMALL_LLM               - true/false, default: false
#   BALANCE_TRAIN               - true/false, default: true
#   BALANCE_DEV                 - true/false, default: false
#   BALANCE_TEST                - true/false, default: false
#   DEV_RATIO                   - default: 0.1
#   TRAIN_MAX_PER_CLASS         - 0 means no cap
#   TEST_MAX_PER_COUNTRY        - 0 means no cap
#
# ConfliBERT variables:
#   CONFLIBERT_BASE_MODEL       - default: models/conflibert (local pre-downloaded model)
#   CONFLIBERT_EPOCHS           - default: 4
#   CONFLIBERT_BATCH_SIZE       - default: 8
#   CONFLIBERT_LR               - default: 5e-5
#   CONFLIBERT_WARMUP_HEAD_EPOCHS - epochs to train head only before full fine-tune (default: 1, set 0 to skip)
#
# Small LLM variables:
#   SMALL_LLM_BASE_MODEL        - required when RUN_SMALL_LLM=true
#                                 Can be a local path (e.g., models/Llama-3.2-3B) or HF model ID
#                                 See lib/core/constants.LOCAL_BASE_MODELS for local model paths
#   SMALL_LLM_MODEL_NAME        - Ollama model name to register and serve the fine-tuned model (auto-resolved from SMALL_LLM_BASE_MODEL)
#   SMALL_LLM_EPOCHS            - default: 3
#   SMALL_LLM_BATCH_SIZE        - default: 4
#   SMALL_LLM_GRAD_ACCUM        - default: 4
#   SMALL_LLM_LR                - default: 2e-4
#
# Examples:
#   Fine-tune Llama-3.2-3B locally: SMALL_LLM_BASE_MODEL=models/Llama-3.2-3B RUN_SMALL_LLM=true ./run_finetuned_baselines.sh
#   Fine-tune HF model directly:    SMALL_LLM_BASE_MODEL=meta-llama/Llama-3.2-1B RUN_SMALL_LLM=true ./run_finetuned_baselines.sh
#
###############################################################################

set -e
set -u
set -o pipefail

SPLIT_VERSION="${SPLIT_VERSION:-}"
EVAL_COUNTRIES="${EVAL_COUNTRIES:-cmr,nga}"
RUN_CONFLIBERT="${RUN_CONFLIBERT:-true}"
RUN_SMALL_LLM="${RUN_SMALL_LLM:-false}"

BALANCE_TRAIN="${BALANCE_TRAIN:-true}"
BALANCE_DEV="${BALANCE_DEV:-false}"
BALANCE_TEST="${BALANCE_TEST:-false}"
DEV_RATIO="${DEV_RATIO:-0.1}"
SEED="${SEED:-42}"
TRAIN_MAX_PER_CLASS="${TRAIN_MAX_PER_CLASS:-0}"
DEV_MAX_PER_CLASS="${DEV_MAX_PER_CLASS:-0}"
TEST_MAX_PER_COUNTRY="${TEST_MAX_PER_COUNTRY:-0}"

CONFLIBERT_BASE_MODEL="${CONFLIBERT_BASE_MODEL:-models/conflibert}"
CONFLIBERT_EPOCHS="${CONFLIBERT_EPOCHS:-4}"
CONFLIBERT_BATCH_SIZE="${CONFLIBERT_BATCH_SIZE:-8}"
CONFLIBERT_LR="${CONFLIBERT_LR:-5e-5}"
CONFLIBERT_WARMUP_HEAD_EPOCHS="${CONFLIBERT_WARMUP_HEAD_EPOCHS:-1}"

SMALL_LLM_BASE_MODEL="${SMALL_LLM_BASE_MODEL:-}"
SMALL_LLM_MODEL_NAME="${SMALL_LLM_MODEL_NAME:-acled-small-llm-ft}"
SMALL_LLM_EPOCHS="${SMALL_LLM_EPOCHS:-3}"
SMALL_LLM_BATCH_SIZE="${SMALL_LLM_BATCH_SIZE:-4}"
SMALL_LLM_GRAD_ACCUM="${SMALL_LLM_GRAD_ACCUM:-4}"
SMALL_LLM_LR="${SMALL_LLM_LR:-2e-4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Ensure Python can import the project package
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Use HuggingFace Hub offline mode to avoid lock contention on shared filesystems (HPC)
export HF_HUB_OFFLINE=1

# Suppress Triton autotune cache writes on Lustre by using node-local temp dir (avoids filelock hangs)
export TRITON_CACHE_DIR=${TMPDIR:-/tmp}

# Use node-local temp dir for datasets library cache to avoid Lustre file-locking issues
export HF_DATASETS_CACHE=${TMPDIR:-/tmp}/hf_datasets_cache
mkdir -p "$HF_DATASETS_CACHE"

# Disable file locking in datasets library (not supported on Lustre)
export DISABLE_FILE_LOCKING=1

# Auto-detect Python executable: prefer .venv if present, else use conda/system python
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    VENV_PY="$REPO_ROOT/.venv/bin/python"
else
    VENV_PY="python"
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_phase() {
    echo -e "\n${BLUE}==================================================================="
    echo -e "$1"
    echo -e "===================================================================${NC}\n"
}
log_step() { echo -e "${GREEN}▶ $1${NC}"; }
log_warn() { echo -e "${YELLOW}⚠ $1${NC}"; }
log_error() { echo -e "${RED}✗ $1${NC}"; }
log_success() { echo -e "${GREEN}✓ $1${NC}"; }
log_info() { echo -e "${CYAN}ℹ $1${NC}"; }

if [ -z "$SPLIT_VERSION" ]; then
    log_error "SPLIT_VERSION is required"
    log_info "Example: SPLIT_VERSION=acled_v1 ./experiments/scripts/run_finetuned_baselines.sh"
    exit 1
fi

if ! command -v "$VENV_PY" &> /dev/null; then
    log_error "Python executable not found: $VENV_PY"
    log_info "Ensure .venv is activated or conda environment is activated"
    exit 1
fi

if [ "$RUN_SMALL_LLM" = "true" ] && [ -z "$SMALL_LLM_BASE_MODEL" ]; then
    log_error "SMALL_LLM_BASE_MODEL must be set when RUN_SMALL_LLM=true"
    log_info "Specify a local path or HF model ID:"
    log_info "  SMALL_LLM_BASE_MODEL=models/Llama-3.2-3B $0"
    log_info "  SMALL_LLM_BASE_MODEL=models/Mistral-7B-v0.3 $0"
    log_info "  SMALL_LLM_BASE_MODEL=models/gemma-3-4b-pt $0"
    log_info "  SMALL_LLM_BASE_MODEL=models/OLMo-2-1124-7B $0"
    log_info "  SMALL_LLM_BASE_MODEL=meta-llama/Llama-2-7b-hf $0  # or any HF model ID"
    exit 1
fi

if [ "$RUN_SMALL_LLM" = "true" ] && [ -n "$SMALL_LLM_BASE_MODEL" ]; then
    # HF_HUB_OFFLINE=1 is always set, so only local model directories are supported.
    if [ ! -d "$SMALL_LLM_BASE_MODEL" ]; then
        log_error "SMALL_LLM_BASE_MODEL path not found: $SMALL_LLM_BASE_MODEL"
        log_error "HF_HUB_OFFLINE=1 is set — only local model directories are supported."
        exit 1
    fi
    log_info "Using local base model: $SMALL_LLM_BASE_MODEL"
fi

cd "$REPO_ROOT"

SPLIT_DIR="data/processed/splits/$SPLIT_VERSION"
TRAIN_CSV="$SPLIT_DIR/train.csv"
DEV_CSV="$SPLIT_DIR/dev.csv"
TEST_CMR="$SPLIT_DIR/test_cmr.csv"
TEST_NGA="$SPLIT_DIR/test_nga.csv"

BASE_RESULTS_DIR="results/baselines/$SPLIT_VERSION"
mkdir -p "$BASE_RESULTS_DIR"

log_phase "PHASE 1: BUILD TRAIN/DEV/TEST SPLITS"

SPLIT_ARGS=(
  --split-version "$SPLIT_VERSION"
  --eval-countries "$EVAL_COUNTRIES"
  --dev-ratio "$DEV_RATIO"
  --seed "$SEED"
  --train-max-per-class "$TRAIN_MAX_PER_CLASS"
  --dev-max-per-class "$DEV_MAX_PER_CLASS"
  --test-max-per-country "$TEST_MAX_PER_COUNTRY"
)

[ "$BALANCE_TRAIN" = "true" ] && SPLIT_ARGS+=(--balance-train)
[ "$BALANCE_DEV" = "true" ] && SPLIT_ARGS+=(--balance-dev)
[ "$BALANCE_TEST" = "true" ] && SPLIT_ARGS+=(--balance-test)

log_step "Creating split bundle: $SPLIT_VERSION"
"$VENV_PY" experiments/pipelines/data/build_train_dev_test_splits.py "${SPLIT_ARGS[@]}"
log_success "Split bundle ready at $SPLIT_DIR"

if [ "$RUN_CONFLIBERT" = "true" ]; then
    log_phase "PHASE 2: FINE-TUNE CONFLIBERT"

    CONFLIBERT_TAG="${SPLIT_VERSION}_seed${SEED}"
    CONFLIBERT_MODEL_DIR="models/conflibert_finetuned_${CONFLIBERT_TAG}"

    log_step "Training ConfliBERT baseline"
    "$VENV_PY" experiments/pipelines/conflibert/finetune_conflibert.py \
      --train-csv "$TRAIN_CSV" \
      --val-csv "$DEV_CSV" \
      --model-id "$CONFLIBERT_BASE_MODEL" \
      --tag "$CONFLIBERT_TAG" \
      --out-root models \
      --epochs "$CONFLIBERT_EPOCHS" \
      --per-device-train-batch-size "$CONFLIBERT_BATCH_SIZE" \
      --learning-rate "$CONFLIBERT_LR" \
      --warmup-head-epochs "$CONFLIBERT_WARMUP_HEAD_EPOCHS"

    log_phase "PHASE 3: EVALUATE CONFLIBERT ON HELD-OUT COUNTRIES"
    CONFLI_RESULTS_DIR="$BASE_RESULTS_DIR/conflibert"
    mkdir -p "$CONFLI_RESULTS_DIR"

    if [ -f "$TEST_CMR" ]; then
      log_step "Evaluating ConfliBERT on Cameroon split"
      "$VENV_PY" experiments/pipelines/conflibert/evaluate_conflibert_on_split.py \
        --model-path "$CONFLIBERT_MODEL_DIR" \
        --input-csv "$TEST_CMR" \
        --output-csv "$CONFLI_RESULTS_DIR/conflibert_predictions_cmr.csv" \
        --model-name "conflibert_finetuned_${CONFLIBERT_TAG}"
    fi

    if [ -f "$TEST_NGA" ]; then
      log_step "Evaluating ConfliBERT on Nigeria split"
      "$VENV_PY" experiments/pipelines/conflibert/evaluate_conflibert_on_split.py \
        --model-path "$CONFLIBERT_MODEL_DIR" \
        --input-csv "$TEST_NGA" \
        --output-csv "$CONFLI_RESULTS_DIR/conflibert_predictions_nga.csv" \
        --model-name "conflibert_finetuned_${CONFLIBERT_TAG}"
    fi

    log_success "ConfliBERT baseline complete"
else
    log_warn "Skipping ConfliBERT baseline (RUN_CONFLIBERT=false)"
fi

if [ "$RUN_SMALL_LLM" = "true" ]; then
    log_phase "PHASE 4: PREPARE SFT DATA FOR SMALL LLM"

    SFT_DIR="$BASE_RESULTS_DIR/sft_data"
    mkdir -p "$SFT_DIR"
    TRAIN_JSONL="$SFT_DIR/train.jsonl"
    DEV_JSONL="$SFT_DIR/dev.jsonl"

    "$VENV_PY" experiments/pipelines/ollama/prepare_sft_data.py \
      --input-csv "$TRAIN_CSV" \
      --output-jsonl "$TRAIN_JSONL"

    "$VENV_PY" experiments/pipelines/ollama/prepare_sft_data.py \
      --input-csv "$DEV_CSV" \
      --output-jsonl "$DEV_JSONL"

    log_phase "PHASE 5: FINE-TUNE SMALL LLM (LORA SFT)"

    SMALL_LLM_ADAPTER_DIR="models/small_llm_adapter_${SPLIT_VERSION}_seed${SEED}"
    SMALL_LLM_MERGED_DIR="models/small_llm_merged_${SPLIT_VERSION}_seed${SEED}"

    FINETUNE_ARGS=(
      --train-jsonl "$TRAIN_JSONL"
      --dev-jsonl "$DEV_JSONL"
      --base-model "$SMALL_LLM_BASE_MODEL"
      --output-dir "$SMALL_LLM_ADAPTER_DIR"
      --merged-output-dir "$SMALL_LLM_MERGED_DIR"
      --epochs "$SMALL_LLM_EPOCHS"
      --batch-size "$SMALL_LLM_BATCH_SIZE"
      --grad-accum "$SMALL_LLM_GRAD_ACCUM"
      --learning-rate "$SMALL_LLM_LR"
      --create-ollama-model
      --ollama-model-name "$SMALL_LLM_MODEL_NAME"
    )

    "$VENV_PY" experiments/pipelines/ollama/finetune_small_llm.py "${FINETUNE_ARGS[@]}"

    log_success "LoRA fine-tuning complete"
    log_info "Adapter saved:  $SMALL_LLM_ADAPTER_DIR"
    log_info "Merged model:   $SMALL_LLM_MERGED_DIR"
    log_info "Ollama model:   $SMALL_LLM_MODEL_NAME"

    log_phase "PHASE 6: EVALUATE SMALL LLM ON HELD-OUT COUNTRIES"
    SMALL_RESULTS_DIR="$BASE_RESULTS_DIR/small_llm"
    mkdir -p "$SMALL_RESULTS_DIR"

    if [ -f "$TEST_CMR" ]; then
      "$VENV_PY" experiments/pipelines/ollama/evaluate_ollama_on_split.py \
        --model "$SMALL_LLM_MODEL_NAME" \
        --input-csv "$TEST_CMR" \
        --output-csv "$SMALL_RESULTS_DIR/ollama_predictions_cmr.csv"
    fi

    if [ -f "$TEST_NGA" ]; then
      "$VENV_PY" experiments/pipelines/ollama/evaluate_ollama_on_split.py \
        --model "$SMALL_LLM_MODEL_NAME" \
        --input-csv "$TEST_NGA" \
        --output-csv "$SMALL_RESULTS_DIR/ollama_predictions_nga.csv"
    fi

    log_success "Small-LLM baseline complete"
else
    log_warn "Skipping small-LLM baseline (RUN_SMALL_LLM=false)"
fi

log_phase "PHASE 7: GENERATE COMPARISON TABLES"
log_step "Building reviewer-ready baseline comparison tables"
"$VENV_PY" -m lib.analysis.compare_finetuned_baselines \
  --split-version "$SPLIT_VERSION" \
  --results-root "results/baselines"

log_success "Comparison tables generated"

log_phase "BASELINE PIPELINE COMPLETE"
log_info "Split bundle:        $SPLIT_DIR"
log_info "Baseline results:    $BASE_RESULTS_DIR"
log_info "Manifest:            $SPLIT_DIR/manifest.json"
log_info "Core metrics:        $BASE_RESULTS_DIR/baseline_core_metrics.csv"
log_info "Publication table:   $BASE_RESULTS_DIR/baseline_publication_table.csv"

if [ "$RUN_CONFLIBERT" = "true" ]; then
  log_info "ConfliBERT outputs:  $BASE_RESULTS_DIR/conflibert/"
fi
if [ "$RUN_SMALL_LLM" = "true" ]; then
  log_info "Small LLM outputs:   $BASE_RESULTS_DIR/small_llm/"
  log_info "Merged HF model:     models/small_llm_merged_${SPLIT_VERSION}_seed${SEED}/"
fi
