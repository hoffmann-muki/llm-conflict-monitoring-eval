#!/bin/bash
###############################################################################
# Full Analysis Pipeline for LLM State Actor Bias Research
#
# This script orchestrates the complete analysis workflow from model inference
# through calibration, fairness metrics, harm analysis, and counterfactual
# perturbation testing.
#
# WORKFLOW: Per-Model-Then-Aggregate
# ----------------------------------
# 1. Run inference one model at a time (saves disk space)
# 2. Each model produces: ollama_results_{model-slug}_acled_{country}_state_actors.csv
# 3. The sample file is created once and reused for all models (fair comparison)
# 4. Before analysis phases, aggregator combines per-model files into one
# 5. Analysis phases produce both combined and per-model output files
#
# Usage:
#   # Run inference with one model
#   COUNTRY=cmr SAMPLE_SIZE=500 STRATEGY=zero_shot OLLAMA_MODELS="mistral:7b" ./scripts/run_full_analysis.sh
#
#   # Run inference with another model (reuses same sample)
#   COUNTRY=cmr SAMPLE_SIZE=500 OLLAMA_MODELS="llama3.1:8b" SKIP_SAMPLING=true ./scripts/run_full_analysis.sh
#
#   # Skip inference, just run analysis on existing per-model results
#   COUNTRY=cmr SAMPLE_SIZE=500 SKIP_INFERENCE=true ./scripts/run_full_analysis.sh
#
# Environment Variables:
#   STRATEGY             - Prompting strategy (zero_shot, few_shot, explainable) [default: zero_shot]
#   NUM_EXAMPLES         - Number of few-shot examples (1-5), only for few_shot strategy [default: 3]
#   COUNTRY              - Country code (cmr, nga) [default: cmr]
#   SAMPLE_SIZE          - Number of events to sample [default: 500]
#   OLLAMA_MODELS        - Models for inference (single or comma-separated) [default: all WORKING_MODELS]
#   CF_MODELS            - Models for counterfactual analysis [default: all WORKING_MODELS]
#   CF_EVENTS            - Number of events for counterfactual [default: 50]
#   SKIP_INFERENCE       - Skip phase 1 if predictions exist [default: false]
#   SKIP_COUNTERFACTUAL  - Skip counterfactual analysis [default: false]
#
# Directory Structure:
#   results/{country}/{strategy}/{sample_size}/              (for zero_shot, explainable)
#   results/{country}/few_shot/{sample_size}/{num_examples}/ (for few_shot)
#     ├── ollama_results_*_acled_{country}_state_actors.csv  (per-model)
#     ├── ollama_results_acled_{country}_state_actors.csv    (combined)
#     ├── ollama_results_calibrated.csv
#     └── ... (metrics, harm, counterfactual outputs)
#
###############################################################################

set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Treat unset variables as an error
set -o pipefail  # Pipeline fails if any command fails

# Configuration with sensible defaults
STRATEGY="${STRATEGY:-zero_shot}"
NUM_EXAMPLES="${NUM_EXAMPLES:-3}"
COUNTRY="${COUNTRY:-cmr}"
SAMPLE_SIZE="${SAMPLE_SIZE:-500}"
OLLAMA_MODELS="${OLLAMA_MODELS:-}"
CF_MODELS="${CF_MODELS:-}"
CF_EVENTS="${CF_EVENTS:-50}"
SKIP_INFERENCE="${SKIP_INFERENCE:-false}"
SKIP_COUNTERFACTUAL="${SKIP_COUNTERFACTUAL:-false}"

# Derived paths - add num_examples subdirectory for few_shot strategy
if [ "$STRATEGY" = "few_shot" ]; then
    RESULTS_DIR="results/${COUNTRY}/${STRATEGY}/${SAMPLE_SIZE}/${NUM_EXAMPLES}"
else
    RESULTS_DIR="results/${COUNTRY}/${STRATEGY}/${SAMPLE_SIZE}"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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
    echo -e "${YELLOW}⚠ Warning: $1${NC}"
}

log_error() {
    echo -e "${RED}✗ Error: $1${NC}"
}

log_success() {
    echo -e "${GREEN} $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log_step "Checking prerequisites..."
    
    # Check Python is available
    if ! command -v python &> /dev/null; then
        log_error "Python not found. Please ensure Python 3.7+ is installed."
        exit 1
    fi
    
    # Check if we're in the right directory
    if [ ! -f "experiments/pipelines/ollama/run_ollama_classification.py" ]; then
        log_error "Must be run from repository root directory"
        exit 1
    fi
    
    # Check if results directory exists (includes strategy and sample_size subdirectories)
    mkdir -p "${RESULTS_DIR}"
    
    log_success "Prerequisites check passed"
}

# Phase 1: Model Inference
run_inference() {
    if [ "$SKIP_INFERENCE" = "true" ]; then
        log_warn "Skipping inference phase (SKIP_INFERENCE=true)"
        
        # Check if any per-model results exist (in strategy/sample_size subdirectory)
        PER_MODEL_COUNT=$(find "${RESULTS_DIR}" -name "ollama_results_*_acled_${COUNTRY}_state_actors.csv" -type f 2>/dev/null | wc -l)
        if [ "$PER_MODEL_COUNT" -eq 0 ]; then
            log_error "No per-model prediction files found. Cannot skip inference."
            exit 1
        fi
        log_success "Found $PER_MODEL_COUNT per-model result file(s)"
        return 0
    fi
    
    log_phase "[Phase 1/5] Model Inference - Generating Predictions ($STRATEGY strategy, $SAMPLE_SIZE samples)"
    log_step "Running classification pipeline for country: ${COUNTRY}, sample size: ${SAMPLE_SIZE}, strategy: ${STRATEGY}"
    
    # Note: If sample file exists, it will be reused for cross-model consistency
    if [ -f "datasets/${COUNTRY}/state_actor_sample_${COUNTRY}_${SAMPLE_SIZE}.csv" ]; then
        log_step "Existing sample file found - will be reused for fair cross-model comparison"
    fi
    
    # Set OLLAMA_MODELS if provided, otherwise will use WORKING_MODELS from constants
    if [ -n "$OLLAMA_MODELS" ]; then
        log_step "Running inference with model(s): ${OLLAMA_MODELS}"
        STRATEGY="${STRATEGY}" NUM_EXAMPLES="${NUM_EXAMPLES}" COUNTRY="${COUNTRY}" SAMPLE_SIZE="${SAMPLE_SIZE}" \
            OLLAMA_MODELS="${OLLAMA_MODELS}" \
            "${VENV_PY:-python}" experiments/pipelines/ollama/run_ollama_classification.py
    else
        log_step "Running inference with all WORKING_MODELS"
        STRATEGY="${STRATEGY}" NUM_EXAMPLES="${NUM_EXAMPLES}" COUNTRY="${COUNTRY}" SAMPLE_SIZE="${SAMPLE_SIZE}" \
            "${VENV_PY:-python}" experiments/pipelines/ollama/run_ollama_classification.py
    fi
    
    log_success "Phase 1 complete: Per-model predictions generated in ${RESULTS_DIR}/"
}

# Phase 1.5: Aggregate per-model results
run_aggregation() {
    log_phase "[Phase 1.5/5] Aggregating Per-Model Results"
    
    log_step "Scanning for per-model result files in ${RESULTS_DIR}/..."
    PER_MODEL_COUNT=$(find "${RESULTS_DIR}" -name "ollama_results_*_acled_${COUNTRY}_state_actors.csv" -type f 2>/dev/null | wc -l)
    
    if [ "$PER_MODEL_COUNT" -eq 0 ]; then
        log_error "No per-model result files found in ${RESULTS_DIR}/"
        exit 1
    fi
    
    log_step "Found $PER_MODEL_COUNT per-model result file(s). Aggregating..."
    COUNTRY="${COUNTRY}" STRATEGY="${STRATEGY}" SAMPLE_SIZE="${SAMPLE_SIZE}" NUM_EXAMPLES="${NUM_EXAMPLES}" "${VENV_PY:-python}" -m lib.core.result_aggregator
    
    # Verify combined file was created (now in strategy/sample_size subdirectory)
    if [ ! -f "${RESULTS_DIR}/ollama_results_acled_${COUNTRY}_state_actors.csv" ]; then
        log_error "Aggregation failed - combined results file not created"
        exit 1
    fi
    
    log_success "Phase 1.5 complete: Results aggregated for cross-model analysis"
}

# Phase 2: Calibration and Core Metrics
run_calibration_and_metrics() {
    log_phase "[Phase 2/5] Calibration & Core Metrics"
    
    log_step "Applying calibration and computing Brier scores..."
    COUNTRY="${COUNTRY}" STRATEGY="${STRATEGY}" SAMPLE_SIZE="${SAMPLE_SIZE}" NUM_EXAMPLES="${NUM_EXAMPLES}" "${VENV_PY:-python}" -m lib.analysis.calibration
    log_success "Calibration complete"
    
    log_step "Computing classification metrics, fairness metrics, and error correlations..."
    COUNTRY="${COUNTRY}" STRATEGY="${STRATEGY}" SAMPLE_SIZE="${SAMPLE_SIZE}" NUM_EXAMPLES="${NUM_EXAMPLES}" "${VENV_PY:-python}" -m lib.analysis.metrics
    log_success "Core metrics computed"
    
    log_step "Computing per-class decision thresholds..."
    COUNTRY="${COUNTRY}" STRATEGY="${STRATEGY}" SAMPLE_SIZE="${SAMPLE_SIZE}" NUM_EXAMPLES="${NUM_EXAMPLES}" "${VENV_PY:-python}" -m lib.analysis.thresholds
    log_success "Thresholds computed"
    
    log_success "Phase 2 complete: Calibration and core metrics"
}

# Phase 3: Bias and Harm Analysis
run_bias_and_harm_analysis() {
    log_phase "[Phase 3/5] Bias & Harm Analysis"
    
    log_step "Computing False Legitimization/Illegitimization rates..."
    COUNTRY="${COUNTRY}" STRATEGY="${STRATEGY}" SAMPLE_SIZE="${SAMPLE_SIZE}" NUM_EXAMPLES="${NUM_EXAMPLES}" "${VENV_PY:-python}" -m lib.analysis.harm
    log_success "Harm metrics computed"
    
    log_step "Generating per-class reports and sampling error cases..."
    COUNTRY="${COUNTRY}" STRATEGY="${STRATEGY}" SAMPLE_SIZE="${SAMPLE_SIZE}" NUM_EXAMPLES="${NUM_EXAMPLES}" "${VENV_PY:-python}" -m lib.analysis.per_class_metrics
    log_success "Error case sampling complete"
    
    log_step "Creating visualization plots..."
    COUNTRY="${COUNTRY}" STRATEGY="${STRATEGY}" SAMPLE_SIZE="${SAMPLE_SIZE}" NUM_EXAMPLES="${NUM_EXAMPLES}" "${VENV_PY:-python}" -m lib.analysis.visualize_reports
    log_success "Visualizations generated"
    
    log_success "Phase 3 complete: Bias and harm analysis"
}

# Phase 4: Counterfactual Analysis
run_counterfactual_analysis() {
    if [ "$SKIP_COUNTERFACTUAL" = "true" ]; then
        log_warn "Skipping counterfactual analysis (SKIP_COUNTERFACTUAL=true)"
        return 0
    fi
    
    log_phase "[Phase 4/5] Counterfactual Perturbation Analysis"
    
    # If CF_MODELS not set, use all WORKING_MODELS from constants
    if [ -n "$CF_MODELS" ]; then
        log_step "Running counterfactual analysis with models: ${CF_MODELS}"
        log_step "Testing ${CF_EVENTS} top disagreement events with hypothesis-driven perturbations..."
        
        if COUNTRY="${COUNTRY}" STRATEGY="${STRATEGY}" SAMPLE_SIZE="${SAMPLE_SIZE}" NUM_EXAMPLES="${NUM_EXAMPLES}" "${VENV_PY:-python}" -m lib.analysis.counterfactual \
            --models "${CF_MODELS}" \
            --events "${CF_EVENTS}"; then
            CF_ANALYSIS_SUCCESS=true
        else
            CF_ANALYSIS_SUCCESS=false
        fi
    else
        log_step "Running counterfactual analysis with all WORKING_MODELS"
        log_step "Testing ${CF_EVENTS} top disagreement events with hypothesis-driven perturbations..."
        
        if COUNTRY="${COUNTRY}" STRATEGY="${STRATEGY}" SAMPLE_SIZE="${SAMPLE_SIZE}" NUM_EXAMPLES="${NUM_EXAMPLES}" "${VENV_PY:-python}" -m lib.analysis.counterfactual \
            --events "${CF_EVENTS}"; then
            CF_ANALYSIS_SUCCESS=true
        else
            CF_ANALYSIS_SUCCESS=false
        fi
    fi
    
    if [ "$CF_ANALYSIS_SUCCESS" = true ]; then
        log_success "Counterfactual analysis complete"
        
        # Find the most recent counterfactual output file (now in strategy/sample_size subdirectory)
        CF_FILE=$(find "${RESULTS_DIR}" -name "counterfactual_analysis_*.json" -type f -print0 2>/dev/null | \
                  xargs -0 ls -t 2>/dev/null | head -1)
        
        if [ -n "$CF_FILE" ] && [ -f "$CF_FILE" ]; then
            log_step "Visualizing counterfactual results..."
            
            if COUNTRY="${COUNTRY}" STRATEGY="${STRATEGY}" SAMPLE_SIZE="${SAMPLE_SIZE}" NUM_EXAMPLES="${NUM_EXAMPLES}" "${VENV_PY:-python}" -m lib.analysis.visualize_counterfactual --input "$CF_FILE"; then
                log_success "Counterfactual visualizations generated"
            else
                log_warn "Counterfactual visualization failed (non-critical)"
            fi
        else
            log_warn "Counterfactual output file not found"
        fi
    else
        log_warn "Counterfactual analysis failed (non-critical)"
    fi
    
    log_success "Phase 4 complete: Counterfactual analysis"
}

# Phase 5: Generate Summary Report
generate_summary() {
    log_phase "[Phase 5/5] Analysis Complete - Summary Report"
    
    echo ""
    echo "Strategy: ${STRATEGY}"
    if [ "$STRATEGY" = "few_shot" ]; then
        echo "Few-shot Examples: ${NUM_EXAMPLES}"
    fi
    echo "Country: ${COUNTRY}"
    echo "Sample Size: ${SAMPLE_SIZE}"
    echo "Results Directory: ${RESULTS_DIR}/"
    echo ""
    echo "====================================================================="
    echo "GENERATED OUTPUTS"
    echo "====================================================================="
    echo ""
    
    # Per-model inference results
    echo "📁 Per-Model Inference Results:"
    for f in "${RESULTS_DIR}"/ollama_results_*_acled_${COUNTRY}_state_actors.csv; do
        [ -f "$f" ] && echo "  ✓ $(basename "$f")"
    done
    
    # Core outputs
    echo ""
    echo "📊 Core Predictions & Calibration (combined + per-model):"
    [ -f "${RESULTS_DIR}/ollama_results_acled_${COUNTRY}_state_actors.csv" ] && \
        echo " ollama_results_acled_${COUNTRY}_state_actors.csv (raw predictions)"
    [ -f "${RESULTS_DIR}/ollama_results_calibrated.csv" ] && \
        echo " ollama_results_calibrated.csv (calibrated predictions)"
    [ -f "${RESULTS_DIR}/calibration_brier_scores.csv" ] && \
        echo " calibration_brier_scores.csv (Brier score analysis)"
    [ -f "${RESULTS_DIR}/reliability_diagrams.png" ] && \
        echo " reliability_diagrams.png (calibration plots)"
    
    echo ""
    echo "Classification Metrics:"
    [ -f "${RESULTS_DIR}/metrics_acled_${COUNTRY}_state_actors.csv" ] && \
        echo " metrics_acled_${COUNTRY}_state_actors.csv (P/R/F1, accuracy)"
    [ -f "${RESULTS_DIR}/confusion_matrices_acled_${COUNTRY}_state_actors.json" ] && \
        echo " confusion_matrices_acled_${COUNTRY}_state_actors.json"
    [ -f "${RESULTS_DIR}/per_class_report.csv" ] && \
        echo " per_class_report.csv (per-class performance)"
    
    echo ""
    echo "Fairness & Bias Metrics:"
    [ -f "${RESULTS_DIR}/fairness_metrics_acled_${COUNTRY}_state_actors.csv" ] && \
        echo " fairness_metrics_acled_${COUNTRY}_state_actors.csv (SPD, Equalized Odds)"
    [ -f "${RESULTS_DIR}/harm_metrics_detailed.csv" ] && \
        echo " harm_metrics_detailed.csv (False Legitimization/Illegitimization rates)"
    [ -f "${RESULTS_DIR}/fl_fi_by_model.csv" ] && \
        echo " fl_fi_by_model.csv (FL/FI counts)"
    
    echo ""
    echo "Source & Error Analysis:"
    [ -f "${RESULTS_DIR}/error_correlations_acled_${COUNTRY}_state_actors.csv" ] && \
        echo " error_correlations_acled_${COUNTRY}_state_actors.csv (notes length correlation)"
    [ -f "${RESULTS_DIR}/top_disagreements.csv" ] && \
        echo " top_disagreements.csv (high-confidence disagreements)"
    [ -f "${RESULTS_DIR}/error_cases_false_legitimization.csv" ] && \
        echo " error_cases_false_legitimization.csv (N≤200 error samples)"
    [ -f "${RESULTS_DIR}/error_cases_false_illegitimization.csv" ] && \
        echo " error_cases_false_illegitimization.csv (N≤200 error samples)"
    
    echo ""
    echo "Counterfactual Analysis:"
    CF_JSON=$(find "${RESULTS_DIR}" -name "counterfactual_analysis_*.json" -type f 2>/dev/null | head -1)
    CF_CSV=$(find "${RESULTS_DIR}" -name "counterfactual_analysis_*_summary.csv" -type f 2>/dev/null | head -1)
    [ -n "$CF_JSON" ] && [ -f "$CF_JSON" ] && \
        echo " $(basename "$CF_JSON") (CFR, CDE, validity metrics)"
    [ -n "$CF_CSV" ] && [ -f "$CF_CSV" ] && \
        echo " $(basename "$CF_CSV") (summary table)"
    
    echo ""
    echo "Visualizations:"
    [ -f "${RESULTS_DIR}/per_class_metrics.png" ] && \
        echo " per_class_metrics.png"
    [ -f "${RESULTS_DIR}/top_disagreements_table.png" ] && \
        echo " top_disagreements_table.png"
    [ -f "${RESULTS_DIR}/accuracy_vs_coverage.png" ] && \
        echo " accuracy_vs_coverage.png"
    
    echo ""
    echo "====================================================================="
    echo "ANALYSIS METRICS COMPUTED"
    echo "====================================================================="
    echo ""
    echo " Standard Classification: Precision, Recall, F1, Accuracy"
    echo " Calibration: Brier scores, reliability diagrams"
    echo " Fairness: Statistical Parity Difference (SPD) with 95% CI"
    echo " Fairness: Equalized Odds (TPR/FPR) with permutation tests"
    echo " Harm: False Legitimization Rate (FLR)"
    echo " Harm: False Illegitimization Rate (FIR)"
    echo " Source: Error correlation with text length"
    echo " Counterfactual: Flip Rate (CFR) per perturbation type"
    echo " Counterfactual: Differential Effect (CDE) with statistical tests"
    echo " Counterfactual: Soft-validity metrics (edit distance, fluency)"
    echo ""
    echo "====================================================================="
    echo ""
    
    log_success "Full analysis pipeline completed successfully!"
    echo ""
}

# Main execution
main() {
    log_phase "LLM State Actor Bias - Full Analysis Pipeline (Per-Model-Then-Aggregate)"
    echo "Configuration:"
    echo "  Strategy: ${STRATEGY}"
    if [ "$STRATEGY" = "few_shot" ]; then
        echo "  Few-shot Examples: ${NUM_EXAMPLES}"
    fi
    echo "  Country: ${COUNTRY}"
    echo "  Sample Size: ${SAMPLE_SIZE}"
    echo "  Results Directory: ${RESULTS_DIR}"
    echo "  Inference Models: ${OLLAMA_MODELS:-all WORKING_MODELS}"
    echo "  Counterfactual Models: ${CF_MODELS:-all WORKING_MODELS}"
    echo "  Counterfactual Events: ${CF_EVENTS}"
    echo ""
    echo "Workflow:"
    echo "  1. Inference produces per-model result files"
    echo "  2. Sample file is reused across models for fair comparison"
    echo "  3. Aggregator combines per-model files before analysis"
    echo "  4. Analysis produces both combined and per-model outputs"
    echo ""
    
    check_prerequisites
    run_inference
    run_aggregation
    run_calibration_and_metrics
    run_bias_and_harm_analysis
    run_counterfactual_analysis
    generate_summary
}

# Run main function
main "$@"
