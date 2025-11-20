#!/bin/bash
###############################################################################
# Experiment Runner with Prompting Strategy Selection
#
# This script runs classification experiments with different prompting strategies
# and generates complete quantitative analysis for comparison.
#
# Usage:
#   STRATEGY=zero_shot COUNTRY=cmr SAMPLE_SIZE=500 ./experiments/scripts/run_experiment.sh
#   STRATEGY=few_shot COUNTRY=nga SAMPLE_SIZE=1000 ./experiments/scripts/run_experiment.sh
#   STRATEGY=explainable COUNTRY=cmr ./experiments/scripts/run_experiment.sh
#
# Environment Variables:
#   STRATEGY             - Prompting strategy (zero_shot, few_shot, explainable) [default: zero_shot]
#   COUNTRY              - Country code (cmr, nga) [default: cmr]
#   SAMPLE_SIZE          - Number of events to sample [default: 500]
#   CF_MODELS            - Models for counterfactual analysis [default: llama3.2,mistral:7b]
#   CF_EVENTS            - Number of events for counterfactual [default: 50]
#   SKIP_INFERENCE       - Skip phase 1 if predictions exist [default: false]
#   SKIP_COUNTERFACTUAL  - Skip counterfactual analysis [default: false]
#
###############################################################################

set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Treat unset variables as an error
set -o pipefail  # Pipeline fails if any command fails

# Configuration with sensible defaults
STRATEGY="${STRATEGY:-zero_shot}"
COUNTRY="${COUNTRY:-cmr}"
SAMPLE_SIZE="${SAMPLE_SIZE:-500}"
CF_MODELS="${CF_MODELS:-llama3.2,mistral:7b}"
CF_EVENTS="${CF_EVENTS:-50}"
SKIP_INFERENCE="${SKIP_INFERENCE:-false}"
SKIP_COUNTERFACTUAL="${SKIP_COUNTERFACTUAL:-false}"

# Determine repository root (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python"

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

if [ ! -x "$VENV_PY" ]; then
    log_error "Virtual environment python not found at $VENV_PY"
    exit 1
fi
log_success "Python environment: $VENV_PY"

if ! command -v ollama &> /dev/null; then
    log_warn "Ollama command not found. Ensure Ollama is running."
else
    log_success "Ollama found"
fi

# Display configuration
log_phase "EXPERIMENT CONFIGURATION"
log_info "Strategy:           $STRATEGY"
log_info "Country:            $COUNTRY"
log_info "Sample Size:        $SAMPLE_SIZE"
log_info "Skip Inference:     $SKIP_INFERENCE"
log_info "Skip Counterfactual: $SKIP_COUNTERFACTUAL"
log_info "CF Models:          $CF_MODELS"
log_info "CF Events:          $CF_EVENTS"
log_info "Results Directory:  results/$COUNTRY/$STRATEGY/"
echo ""

cd "$REPO_ROOT"

# Phase 1: Model Inference with Strategy
if [ "$SKIP_INFERENCE" = "true" ]; then
    log_phase "PHASE 1: MODEL INFERENCE (SKIPPED)"
    log_warn "Skipping inference - using existing predictions"
else
    log_phase "PHASE 1: MODEL INFERENCE ($STRATEGY strategy)"
    
    log_step "Running classification with $STRATEGY prompting..."
    STRATEGY="$STRATEGY" COUNTRY="$COUNTRY" SAMPLE_SIZE="$SAMPLE_SIZE" \
        "$VENV_PY" experiments/pipelines/run_classification.py
    
    log_success "Inference completed with $STRATEGY strategy"
fi

# Phase 2: Calibration & Core Metrics
log_phase "PHASE 2: CALIBRATION & CORE METRICS"

# Set strategy-specific results path
STRATEGY_RESULTS="results/$COUNTRY/$STRATEGY"
export RESULTS_DIR="$STRATEGY_RESULTS"

log_step "Applying calibration (isotonic + temperature scaling)..."
COUNTRY="$COUNTRY" RESULTS_DIR="$STRATEGY_RESULTS" \
    "$VENV_PY" -m lib.analysis.calibration

log_step "Computing classification metrics and fairness analysis..."
COUNTRY="$COUNTRY" RESULTS_DIR="$STRATEGY_RESULTS" \
    "$VENV_PY" -m lib.analysis.metrics

log_step "Computing per-class decision thresholds..."
COUNTRY="$COUNTRY" RESULTS_DIR="$STRATEGY_RESULTS" \
    "$VENV_PY" -m lib.analysis.thresholds

log_success "Calibration and metrics completed"

# Phase 3: Bias & Harm Analysis
log_phase "PHASE 3: BIAS & HARM ANALYSIS"

log_step "Computing false legitimization/illegitimization rates..."
COUNTRY="$COUNTRY" RESULTS_DIR="$STRATEGY_RESULTS" \
    "$VENV_PY" -m lib.analysis.harm

log_step "Generating per-class metrics and error case sampling..."
COUNTRY="$COUNTRY" RESULTS_DIR="$STRATEGY_RESULTS" \
    "$VENV_PY" -m lib.analysis.per_class_metrics

log_step "Creating visualization plots..."
COUNTRY="$COUNTRY" RESULTS_DIR="$STRATEGY_RESULTS" \
    "$VENV_PY" -m lib.analysis.visualize_reports

log_success "Bias and harm analysis completed"

# Phase 4: Counterfactual Analysis (Optional)
if [ "$SKIP_COUNTERFACTUAL" = "true" ]; then
    log_phase "PHASE 4: COUNTERFACTUAL ANALYSIS (SKIPPED)"
    log_warn "Skipping counterfactual analysis"
else
    log_phase "PHASE 4: COUNTERFACTUAL PERTURBATION ANALYSIS"
    
    log_step "Running counterfactual perturbation testing..."
    COUNTRY="$COUNTRY" RESULTS_DIR="$STRATEGY_RESULTS" \
        "$VENV_PY" -m lib.analysis.counterfactual \
        --models "$CF_MODELS" --events "$CF_EVENTS"
    
    log_step "Generating counterfactual visualizations..."
    COUNTRY="$COUNTRY" RESULTS_DIR="$STRATEGY_RESULTS" \
        "$VENV_PY" -m lib.analysis.visualize_counterfactual \
        --input "$STRATEGY_RESULTS/counterfactual_analysis_${CF_MODELS//,/_}.json"
    
    log_success "Counterfactual analysis completed"
fi

# Phase 5: Summary Report
log_phase "PHASE 5: EXPERIMENT SUMMARY"

echo -e "${CYAN}Strategy: $STRATEGY${NC}"
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

check_file "$STRATEGY_RESULTS/ollama_results_acled_${COUNTRY}_state_actors.csv"
check_file "$STRATEGY_RESULTS/ollama_results_calibrated.csv"
check_file "$STRATEGY_RESULTS/calibration_brier_scores.csv"
check_file "$STRATEGY_RESULTS/metrics_acled_${COUNTRY}_state_actors.csv"
check_file "$STRATEGY_RESULTS/fairness_metrics_acled_${COUNTRY}_state_actors.csv"
check_file "$STRATEGY_RESULTS/harm_metrics_detailed.csv"
check_file "$STRATEGY_RESULTS/error_cases_false_legitimization.csv"
check_file "$STRATEGY_RESULTS/error_cases_false_illegitimization.csv"
check_file "$STRATEGY_RESULTS/error_correlations_acled_${COUNTRY}_state_actors.csv"

if [ "$SKIP_COUNTERFACTUAL" = "false" ]; then
    check_file "$STRATEGY_RESULTS/counterfactual_analysis_${CF_MODELS//,/_}.json"
fi

echo ""
log_phase "EXPERIMENT COMPLETED SUCCESSFULLY"
log_success "All quantitative metrics generated for $STRATEGY strategy"
log_info "Compare with other strategies by running:"
log_info "  STRATEGY=few_shot COUNTRY=$COUNTRY ./experiments/scripts/run_experiment.sh"
log_info "  STRATEGY=explainable COUNTRY=$COUNTRY ./experiments/scripts/run_experiment.sh"
echo ""
