#!/bin/bash
###############################################################################
# ConfliBERT Experiment Runner with Prompting Strategy Selection
#
# This script runs ConfliBERT classification experiments with different
# prompting strategies and generates complete quantitative analysis for
# comparison with Ollama models.
#
# Usage:
#   STRATEGY=zero_shot COUNTRY=cmr SAMPLE_SIZE=500 ./experiments/scripts/run_conflibert_experiment.sh
#   STRATEGY=few_shot NUM_EXAMPLES=3 COUNTRY=nga SAMPLE_SIZE=1000 ./experiments/scripts/run_conflibert_experiment.sh
#   STRATEGY=explainable COUNTRY=cmr ./experiments/scripts/run_conflibert_experiment.sh
#
# Environment Variables:
#   MODEL_PATH           - Path to local ConfliBERT model directory [default: models/conflibert]
#   STRATEGY             - Prompting strategy (zero_shot, few_shot, explainable) [default: zero_shot]
#   NUM_EXAMPLES         - Number of few-shot examples (1-5), only for few_shot strategy [default: 3]
#   COUNTRY              - Country code (cmr, nga) [default: cmr]
#   SAMPLE_SIZE          - Number of events to sample [default: 500]
#   PRIMARY_GROUP        - Event type to oversample [default: none (proportional)]
#   PRIMARY_SHARE        - Fraction for primary group (0-1) [default: 0.0]
#   BATCH_SIZE           - Batch size for inference [default: 16]
#   MAX_LENGTH           - Maximum sequence length [default: 256]
#   DEVICE               - Device (cuda, mps, cpu) [default: auto]
#   SKIP_INFERENCE       - Skip phase 1 if predictions exist [default: false]
#   SKIP_COUNTERFACTUAL  - Skip counterfactual analysis [default: false]
#   CF_EVENTS            - Number of events for counterfactual [default: 20]
#
###############################################################################

set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Treat unset variables as an error
set -o pipefail  # Pipeline fails if any command fails

# Configuration with sensible defaults
MODEL_PATH="${MODEL_PATH:-models/conflibert}"
# ConfliBERT is a supervised, fine-tuned classifier. We run it only as a
# supervised baseline (zero-shot equivalent in our evaluation matrix).
# Force strategy to zero_shot to avoid mixing prompting modalities.
STRATEGY="zero_shot"
NUM_EXAMPLES="${NUM_EXAMPLES:-3}"
COUNTRY="${COUNTRY:-cmr}"
SAMPLE_SIZE="${SAMPLE_SIZE:-500}"
PRIMARY_GROUP="${PRIMARY_GROUP:-}"
PRIMARY_SHARE="${PRIMARY_SHARE:-0.0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-256}"
DEVICE="${DEVICE:-auto}"
SKIP_INFERENCE="${SKIP_INFERENCE:-false}"
SKIP_COUNTERFACTUAL="${SKIP_COUNTERFACTUAL:-false}"
CF_EVENTS="${CF_EVENTS:-20}"

# Determine repository root (two levels up from this script)
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

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_phase() {
    echo -e "\n${BLUE}==================================================================="
    echo -e "$1"
    echo -e "===================================================================${NC}\n"
}

log_step() {
    echo -e "${GREEN}▶ $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

log_error() {
    echo -e "${RED}✗ $1${NC}"
}

log_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

log_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

# Validate strategy
case "$STRATEGY" in
    zero_shot|few_shot|explainable)
        ;;
    *)
        log_error "Invalid strategy: $STRATEGY"
        log_info "Valid strategies: zero_shot, few_shot, explainable"
        exit 1
        ;;
esac

# Validate country
case "$COUNTRY" in
    cmr|nga)
        ;;
    *)
        log_error "Invalid country: $COUNTRY"
        log_info "Valid countries: cmr, nga"
        exit 1
        ;;
esac

# Check prerequisites
log_phase "CHECKING PREREQUISITES"

if ! command -v "$VENV_PY" &> /dev/null; then
    log_error "Python executable not found: $VENV_PY"
    log_info "Ensure .venv is activated or conda environment is activated"
    exit 1
fi
log_success "Python environment: $VENV_PY"

# Check PyTorch and transformers
if ! "$VENV_PY" -c "import torch, transformers" 2>/dev/null; then
    log_error "Required packages (torch, transformers) not found"
    log_info "Install with: pip install torch transformers"
    exit 1
fi
log_success "PyTorch and transformers available"

# Check model path exists unless we're explicitly skipping inference
if [ "$SKIP_INFERENCE" = "true" ]; then
    log_warn "SKIP_INFERENCE=true; skipping model path existence check"
else
    if [ ! -d "$MODEL_PATH" ]; then
        log_error "Model path not found: $MODEL_PATH"
        log_info "Download model with: python experiments/pipelines/conflibert/download_conflibert_model.py --out-dir $MODEL_PATH"
        exit 1
    fi
    log_success "ConfliBERT model found at: $MODEL_PATH"
fi

# Display configuration
log_phase "CONFLIBERT EXPERIMENT CONFIGURATION"
log_info "Model Path:         $MODEL_PATH"
log_info "Strategy:           $STRATEGY"
if [ "$STRATEGY" = "few_shot" ]; then
    log_info "Few-shot Examples:  $NUM_EXAMPLES"
fi
log_info "Country:            $COUNTRY"
log_info "Sample Size:        $SAMPLE_SIZE"
log_info "Primary Group:      ${PRIMARY_GROUP:-none (proportional sampling)}"
log_info "Primary Share:      $PRIMARY_SHARE"
log_info "Batch Size:         $BATCH_SIZE"
log_info "Max Length:         $MAX_LENGTH"
log_info "Device:             $DEVICE"
log_info "Skip Inference:     $SKIP_INFERENCE"
log_info "Skip Counterfactual: $SKIP_COUNTERFACTUAL"
log_info "CF Events:          $CF_EVENTS"

# Build results directory path (includes num_examples for few_shot)
if [ "$STRATEGY" = "few_shot" ]; then
    STRATEGY_RESULTS="results/$COUNTRY/$STRATEGY/$SAMPLE_SIZE/$NUM_EXAMPLES"
else
    STRATEGY_RESULTS="results/$COUNTRY/$STRATEGY/$SAMPLE_SIZE"
fi
log_info "Results Directory:  $STRATEGY_RESULTS"
echo ""

cd "$REPO_ROOT"

# Phase 1: ConfliBERT Inference with Strategy
if [ "$SKIP_INFERENCE" = "true" ]; then
    log_phase "PHASE 1: CONFLIBERT INFERENCE (SKIPPED)"
    log_warn "Skipping inference - using existing predictions"
else
    log_phase "PHASE 1: CONFLIBERT INFERENCE ($STRATEGY strategy)"
    
    log_step "Running ConfliBERT classification with $STRATEGY prompting..."
    
    # Set device argument
    if [ "$DEVICE" = "auto" ]; then
        DEVICE_ARG=""
    else
        DEVICE_ARG="--device $DEVICE"
    fi
    
    # Set primary group arguments if specified
    if [ -n "$PRIMARY_GROUP" ]; then
        PRIMARY_ARGS="--primary-group \"$PRIMARY_GROUP\" --primary-share $PRIMARY_SHARE"
    else
        PRIMARY_ARGS=""
    fi
    
    # Set num_examples argument for few_shot strategy
    if [ "$STRATEGY" = "few_shot" ]; then
        NUM_EXAMPLES_ARG="--num-examples $NUM_EXAMPLES"
    else
        NUM_EXAMPLES_ARG=""
    fi
    
    eval "\"$VENV_PY\" experiments/pipelines/conflibert/run_conflibert_classification.py \"$COUNTRY\" \
        --model-path \"$MODEL_PATH\" \
        --strategy \"$STRATEGY\" \
        --sample-size \"$SAMPLE_SIZE\" \
        --batch-size \"$BATCH_SIZE\" \
        --max-length \"$MAX_LENGTH\" \
        $DEVICE_ARG $PRIMARY_ARGS $NUM_EXAMPLES_ARG"
    
    log_success "ConfliBERT inference completed with $STRATEGY strategy"
fi

# Phase 2: Calibration & Core Metrics
log_phase "PHASE 2: CALIBRATION & CORE METRICS"

# Set strategy-specific results path
export RESULTS_DIR="$STRATEGY_RESULTS"

# Point analysis modules to ConfliBERT results (instead of default ollama_results_*.csv)
CONFLIBERT_RESULTS_CSV="$STRATEGY_RESULTS/conflibert_results_acled_${COUNTRY}_actors.csv"
export RESULTS_CSV="$CONFLIBERT_RESULTS_CSV"
log_info "Using results file: $RESULTS_CSV"

log_step "Applying calibration (isotonic + temperature scaling)..."
COUNTRY="$COUNTRY" STRATEGY="$STRATEGY" SAMPLE_SIZE="$SAMPLE_SIZE" NUM_EXAMPLES="$NUM_EXAMPLES" \
    # Build a combined raw results CSV so calibration runs on all available models
    # This avoids overwriting an existing multi-model calibrated file when ConfliBERT
    # is run after Ollama models. The combined file will be written to
    # $STRATEGY_RESULTS/ollama_results_acled_${COUNTRY}_actors.csv and then passed
    # to the calibration module via RESULTS_CSV.
    # Export shell variables so the embedded Python can access them via os.environ
    export STRATEGY_RESULTS="$STRATEGY_RESULTS"
    export CONFLIBERT_RESULTS_CSV="$CONFLIBERT_RESULTS_CSV"
    "$VENV_PY" - <<PY
import sys
import os, glob, pandas as pd

results_dir = os.environ.get('STRATEGY_RESULTS')
country = os.environ.get('COUNTRY')
conf_csv = os.environ.get('CONFLIBERT_RESULTS_CSV')
files = set()

# Always consider an existing combined Ollama file if present
combined_candidate = os.path.join(results_dir, f'ollama_results_acled_{country}_actors.csv')
if os.path.exists(combined_candidate):
    files.add(combined_candidate)

# Include ConfliBERT CSV if available
if conf_csv and os.path.exists(conf_csv):
    files.add(conf_csv)

# Include any per-model Ollama raw output files under model subdirectories
per_model = glob.glob(os.path.join(results_dir, '*/ollama_results_*_acled_{}_actors.csv'.format(country)))
for p in per_model:
    if os.path.exists(p):
        files.add(p)

files = sorted(list(files))
if not files:
    print('No raw result files found to combine; using', conf_csv)
    combined_path = conf_csv
else:
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print('Warning: could not read', f, e)
    if not dfs:
        combined_path = conf_csv
    else:
        combined = pd.concat(dfs, ignore_index=True)
        combined_path = os.path.join(results_dir, f'ollama_results_acled_{country}_actors.csv')
        # Drop duplicate rows by event_id + model to prefer the first occurrence
        if 'event_id' in combined.columns and 'model' in combined.columns:
            combined = combined.drop_duplicates(subset=['event_id', 'model'], keep='first')
        combined.to_csv(combined_path, index=False)
        print('Wrote combined raw results to', combined_path)

print('Running calibration on:', combined_path)
os.environ['RESULTS_CSV'] = combined_path
os.execv(sys.executable, [sys.executable, '-m', 'lib.analysis.calibration'])
PY

log_step "Computing classification metrics and fairness analysis..."
COUNTRY="$COUNTRY" STRATEGY="$STRATEGY" SAMPLE_SIZE="$SAMPLE_SIZE" NUM_EXAMPLES="$NUM_EXAMPLES" \
    RESULTS_CSV="$CONFLIBERT_RESULTS_CSV" \
    "$VENV_PY" -m lib.analysis.metrics

log_step "Computing per-class decision thresholds..."
COUNTRY="$COUNTRY" STRATEGY="$STRATEGY" SAMPLE_SIZE="$SAMPLE_SIZE" NUM_EXAMPLES="$NUM_EXAMPLES" \
    "$VENV_PY" -m lib.analysis.thresholds

log_success "Calibration and metrics completed"

# Phase 3: Bias & Harm Analysis
log_phase "PHASE 3: BIAS & HARM ANALYSIS"

log_step "Computing false legitimization/illegitimization rates..."
COUNTRY="$COUNTRY" STRATEGY="$STRATEGY" SAMPLE_SIZE="$SAMPLE_SIZE" NUM_EXAMPLES="$NUM_EXAMPLES" \
    "$VENV_PY" -m lib.analysis.harm

log_step "Generating per-class metrics and error case sampling..."
# Set TOP_N_DISAGREEMENTS to match CF_EVENTS so enough disagreements are available for counterfactual analysis
COUNTRY="$COUNTRY" STRATEGY="$STRATEGY" SAMPLE_SIZE="$SAMPLE_SIZE" NUM_EXAMPLES="$NUM_EXAMPLES" \
    TOP_N_DISAGREEMENTS="$CF_EVENTS" \
    "$VENV_PY" -m lib.analysis.per_class_metrics

log_step "Creating visualization plots..."
COUNTRY="$COUNTRY" STRATEGY="$STRATEGY" SAMPLE_SIZE="$SAMPLE_SIZE" NUM_EXAMPLES="$NUM_EXAMPLES" \
    "$VENV_PY" -m lib.analysis.visualize_reports

log_success "Bias and harm analysis completed"

# Phase 4: Counterfactual Analysis
if [ "$SKIP_COUNTERFACTUAL" = "true" ]; then
    log_phase "PHASE 4: COUNTERFACTUAL PERTURBATION ANALYSIS (SKIPPED)"
    log_warn "Skipping counterfactual analysis"
else
    log_phase "PHASE 4: COUNTERFACTUAL PERTURBATION ANALYSIS"
    
    log_step "Running counterfactual perturbation testing on top-N disagreements..."
    
    # ConfliBERT is the only model in this pipeline, so we always use it for counterfactual analysis.
    # Cross-model comparison (ConfliBERT vs Ollama models) happens at the reporting/visualization level.
    COUNTRY="$COUNTRY" STRATEGY="$STRATEGY" SAMPLE_SIZE="$SAMPLE_SIZE" NUM_EXAMPLES="$NUM_EXAMPLES" \
        "$VENV_PY" -m lib.analysis.counterfactual \
        --models "conflibert" --events "$CF_EVENTS"
    
    log_step "Generating counterfactual visualizations..."
    
    # Find the most recent counterfactual output file
    CF_FILE=$(find "$STRATEGY_RESULTS" -name "counterfactual_analysis_*.json" -type f -print0 2>/dev/null | \
              xargs -0 ls -t 2>/dev/null | head -1)
    
    if [ -n "$CF_FILE" ] && [ -f "$CF_FILE" ]; then
        COUNTRY="$COUNTRY" STRATEGY="$STRATEGY" SAMPLE_SIZE="$SAMPLE_SIZE" NUM_EXAMPLES="$NUM_EXAMPLES" \
            "$VENV_PY" -m lib.analysis.visualize_counterfactual \
            --input "$CF_FILE"
    else
        log_warn "Counterfactual output file not found"
    fi

    # Aggregate word-level impact scores across all perturbation types
    # (including newly added neutral_control baseline).  Non-critical.
    log_step "Aggregating word-level impact scores across perturbation types..."
    if COUNTRY="$COUNTRY" STRATEGY="$STRATEGY" SAMPLE_SIZE="$SAMPLE_SIZE" NUM_EXAMPLES="$NUM_EXAMPLES" \
        "$VENV_PY" -m lib.analysis.aggregate_word_impacts_from_counterfactuals; then
        log_success "Word impact aggregation complete"
        log_step "Visualizing word impacts by perturbation type..."
        if COUNTRY="$COUNTRY" STRATEGY="$STRATEGY" SAMPLE_SIZE="$SAMPLE_SIZE" NUM_EXAMPLES="$NUM_EXAMPLES" \
            "$VENV_PY" -m lib.analysis.visualize_word_impacts_by_perturbation; then
            log_success "Word impact visualizations generated"
        else
            log_warn "Word impact visualization failed (non-critical)"
        fi
    else
        log_warn "Word impact aggregation failed (non-critical)"
    fi

    # Error trace: Layer Integrated Gradients (ConfliBERT).
    # RFC rationale analysis is skipped for encoder-only models.
    # Reads the counterfactual JSON already on disk; non-critical.
    log_step "Running error trace analysis (LIG attribution)..."
    if COUNTRY="$COUNTRY" STRATEGY="$STRATEGY" SAMPLE_SIZE="$SAMPLE_SIZE" NUM_EXAMPLES="$NUM_EXAMPLES" \
        "$VENV_PY" -m lib.analysis.error_trace; then
        log_success "Error trace analysis complete"
    else
        log_warn "Error trace analysis failed (non-critical)"
    fi

    log_success "Counterfactual analysis completed"
fi

# Phase 5: Summary Report
log_phase "PHASE 5: EXPERIMENT SUMMARY"

echo -e "${CYAN}Model: ConfliBERT${NC}"
echo -e "${CYAN}Strategy: $STRATEGY${NC}"
if [ "$STRATEGY" = "few_shot" ]; then
    echo -e "${CYAN}Few-shot Examples: $NUM_EXAMPLES${NC}"
fi
echo -e "${CYAN}Country: $COUNTRY${NC}"
echo -e "${CYAN}Results directory: $STRATEGY_RESULTS${NC}\n"

log_info "Generated outputs:"
echo ""

# Check and list key output files
check_file() {
    if [ -f "$1" ]; then
        log_success "$1"
    else
        log_warn "$1 (not found)"
    fi
}

check_file "$STRATEGY_RESULTS/conflibert_results_acled_${COUNTRY}_actors.csv"
check_file "$STRATEGY_RESULTS/ollama_results_calibrated.csv"
check_file "$STRATEGY_RESULTS/calibration_brier_scores.csv"
check_file "$STRATEGY_RESULTS/metrics_acled_${COUNTRY}_actors.csv"
check_file "$STRATEGY_RESULTS/fairness_metrics_acled_${COUNTRY}_actors.csv"
check_file "$STRATEGY_RESULTS/harm_metrics_detailed.csv"
check_file "$STRATEGY_RESULTS/error_cases_false_legitimization.csv"
check_file "$STRATEGY_RESULTS/error_cases_false_illegitimization.csv"
check_file "$STRATEGY_RESULTS/error_correlations_acled_${COUNTRY}_actors.csv"

if [ "$SKIP_COUNTERFACTUAL" = "false" ]; then
    CF_OUTPUT=$(find "$STRATEGY_RESULTS" -name "counterfactual_analysis_*.json" -type f 2>/dev/null | head -1)
    if [ -n "$CF_OUTPUT" ]; then
        check_file "$CF_OUTPUT"
    fi
    check_file "$STRATEGY_RESULTS/word_impacts.csv"
    check_file "$STRATEGY_RESULTS/fig_word_impact_scatter_by_perturbation.png"
    check_file "$STRATEGY_RESULTS/fig_word_impact_bars_by_perturbation.png"
    check_file "$STRATEGY_RESULTS/error_trace_report.json"
    check_file "$STRATEGY_RESULTS/error_trace_summary.csv"
fi

echo ""
log_phase "CONFLIBERT EXPERIMENT COMPLETED SUCCESSFULLY"
log_success "All quantitative metrics generated for ConfliBERT with $STRATEGY strategy"
log_info "Compare with Ollama models by checking $STRATEGY_RESULTS/"
log_info "Run different strategies:"
log_info "  STRATEGY=few_shot NUM_EXAMPLES=3 COUNTRY=$COUNTRY ./experiments/scripts/run_conflibert_experiment.sh"
log_info "  STRATEGY=explainable COUNTRY=$COUNTRY ./experiments/scripts/run_conflibert_experiment.sh"
echo ""
