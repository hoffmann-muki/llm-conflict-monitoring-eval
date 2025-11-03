# Evaluating State Actor Bias

This repository contains code and data for evaluating potential bias in LLM classification of ACLED event types when the primary actor is a state actor (Nigeria-focused dataset for now). The main script samples a stratified dataset and runs classification using local Ollama models.

## What this project does
- Loads an ACLED Nigeria CSV and filters rows where the primary actor is a Nigerian state actor (military/police).
- Normalizes actor names and selects usable rows (has notes and known event types).
- Builds a 150-row stratified sample focusing 60% on "Violence against civilians" and 40% distributed across other event types.
- Runs classification with locally-hosted models via Ollama and saves model predictions.

## Files of interest
- `political_bias_of_llms.py` — reorganized main script. Reads data, builds the stratified sample, runs classification, and writes outputs.
- `datasets/2014-01-01-2024-12-31-Nigeria.csv` — ACLED CSV.
- `datasets/state_actor_6class_150_test.csv` — generated stratified sample.
- `datasets/ollama_results_acled_nigeria_state_actors.csv` — classification outputs.
- `requirements.txt` — pinned Python dependencies to install in a virtual environment.

## Dependencies
- Python 3.11+ (3.13 used in development environment)
- pip
- virtualenv/venv (recommended)
- git and git-lfs (for large data files)
- Ollama application + `ollama` CLI if you want to run model pulls and classification locally.

Python packages (install inside venv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you do not plan to run the Ollama classification steps, the Python dependencies alone are sufficient to build the sample.

## Running the script
1. Ensure `datasets/2014-01-01-2024-12-31-Nigeria.csv` is present (this repo tracks it via Git LFS).
2. Activate the virtual environment and install dependencies (see above).
3. Start the Ollama desktop app/daemon and pull models (if you want to run classification):

```bash
# Ollama app must be running
ollama pull llama3.2
ollama pull qwen2.5
ollama pull mistral
```

4. Run the main script:

```bash
python political_bias_of_llms.py
```

Outputs will be written to `datasets/state_actor_6class_150_test.csv` and `datasets/ollama_results_acled_nigeria_state_actors.csv`.

## Notes
- Large dataset files are tracked with Git LFS; collaborators need git-lfs installed to clone and fetch data.
- The sampling uses deterministic random_state=42 for reproducibility.
- The script currently caps the output sample at 150 rows or the number of usable rows, whichever is smaller.

