# Evaluating State Actor Bias in LLM Event Classification

This is a research framework for measuring how state-actor-specific conflict events are classified by modern LLMs. We orchestrate the same sample across multiple countries, prompting strategies, and models so that every comparison is reproducible and aligned.

## Highlights
- **Shared sample:** `datasets/{country}/state_actor_sample_{country}_{sample_size}.csv` is built once and reused by every model (Ollama or ConfliBERT) for a given country/sample combination.
- **Structured responses:** JSON schema enforcement plus the `reasoning` field make every inference traceable and machine-readable.
- **Cross-model comparison:** `lib.analysis.compare_all_models` discovers Ollama and ConfliBERT outputs, writes combined metrics/harm/fairness tables, and generates comparison plots under `results/.../comparison/`.

## Repository Layout
| Directory | Focus |
| --- | --- |
| `experiments/` | Pipelines, prompts, and shell scripts for running each inference strategy (see `experiments/README.md`). |
| `lib/` | Reusable analysis helpers, inference clients, and aggregation modules (see `lib/README.md`). |
| `datasets/` | Country-specific extracts (ACLED). |
| `results/` | Organized outputs by country/strategy/sample size, with explicit `comparison/` folders. |
| `tests/` | Validation suite (`tests/README.md`). |

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Ollama daemon
ollama list
# ConfliBERT weights (first-time download)
python experiments/pipelines/conflibert/download_conflibert_model.py --out-dir models/conflibert
```

## Running the Full Analysis
```bash
COUNTRY=cmr SAMPLE_SIZE=500 STRATEGY=zero_shot ./experiments/scripts/run_ollama_full_analysis.sh
```
This shell script runs inference → aggregation → calibration → metrics → counterfactual for Ollama models. After that, run `python -m lib.analysis.compare_all_models` (same env vars) to build the comparison folder alongside the root results.

## Results & Comparison
- **Per-model files:** `results/{country}/{strategy}/{sample_size}/{model_slug}/ollama_results_{model}_acled_{country}_actors.csv` (plus a combined `ollama_results_acled_{country}_actors.csv`).
- **Reasoning:** The `reasoning` column now records explainable prompts and aligns with JSON schema enforcement. Missing reasoning entries are stored as empty strings until materialized by the model.
- **Comparison directory:** `results/.../comparison/` contains `all_models_metrics.csv`, `all_models_harm.csv`, `all_models_fairness.csv`, and visualization PNGs for ConfliBERT vs Ollama comparisons.

## Documentation Links
- [experiments/README.md](experiments/README.md) — Running pipelines and shell scripts plus sampling guidance.
- [lib/README.md](lib/README.md) — API-level helpers, inference clients, aggregation, and analysis modules.
- [experiments/prompting_strategies/README.md](experiments/prompting_strategies/README.md) — Custom strategy creation.
- [tests/README.md](tests/README.md) — Validation and test harness.
