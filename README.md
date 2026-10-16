# Are LLMs Ready for Conflict Monitoring? Empirical Evidence from West Africa

**Hoffmann Muki · Olukunle Owolabi**

ACM Conference on Fairness, Accountability, and Transparency (FAccT), 2026

[arXiv](https://arxiv.org/abs/2605.04177) · [Publication](https://doi.org/10.1145/3805689.3812264) · [Reproduction guide](experiments/README.md) · [Citation](#citation)

Can language models classify conflict events without systematically distorting who is responsible for violence? This repository accompanies our study of six models on ACLED event narratives from Cameroon and Nigeria. We evaluate the direction of classification errors, disparities between actor groups, and sensitivity to lexical framing alongside predictive performance.

**The central finding:** domain adaptation reduces directional error and lexical fragility, but actor-based bias persists. The evidence supports human oversight when using these models for conflict monitoring.

## Evaluation at a glance

| | Scope |
| --- | --- |
| Task | Six-class event classification: violence against civilians, battles, explosions/remote violence, protests, riots, and strategic developments. |
| Open-weight models | Llama 3.2 3B, Mistral 7B, Gemma 3 4B, OLMo 2 7B. |
| Adapted models | AfroConfliBERT and AfroConfliLLAMA, implemented through ConfliBERT fine-tuning and causal-LM LoRA SFT. |
| Evaluation | Classification, confidence calibration, actor fairness, False Legitimation/Illegitimation, counterfactual perturbations, and rationale tracing. |

The code provides both prompt-based comparisons on shared country samples and supervised evaluation on separate country test sets. These protocols use different evaluation populations; the [experiment guide](experiments/README.md#evaluation-protocols) explains how each is constructed.

## Quick start

Use Python 3.11+ and run the following from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
git lfs pull
```

Dataset files are tracked with Git LFS. The source export is `datasets/Africa_lagged_data_up_to-2024-10-24.csv`; country samples and supervised splits are supplied alongside it. Model weights must be obtained separately.

To rebuild the Cameroon zero-shot reports from the saved model predictions:

```bash
COUNTRY=cmr STRATEGY=zero_shot SAMPLE_SIZE=1000 \
SKIP_INFERENCE=true SKIP_COUNTERFACTUAL=true \
  bash experiments/scripts/run_full_analysis.sh
```

The command updates calibration, performance, fairness, and harm artifacts in `results/cmr/zero_shot/1000/`. It requires no running model service. For fresh predictions, few-shot comparisons, or training the adapted baselines, follow the [reproduction guide](experiments/README.md).

## Results and analysis

Prediction CSVs, derived tables, and figures are retained under `results/`. Begin with the country reports or the standalone directional-error tables:

| Resource | Cameroon | Nigeria |
| --- | --- | --- |
| Zero-shot predictions and diagnostics | [Browse](results/cmr/zero_shot/1000/) | [Browse](results/nga/zero_shot/1000/) |
| FL/FI tables across conditions | [Browse](results/analysis/fl_fi/cmr/) | [Browse](results/analysis/fl_fi/nga/) |

Report interpretation depends on the evaluation subset and metric definition. In particular, the standalone FL/FI tables count `V ↔ B` errors, while the pipeline harm analysis includes `S` with `B`. See the [metric reference](results/analysis/fl_fi/README.md) for denominators and statistical conventions, and the [analysis reference](lib/README.md) for how reports are produced.

## Working with the code

- **[Run experiments](experiments/README.md):** inference, domain adaptation, fixed-split evaluation, and experiment controls.
- **[Use the analysis modules](lib/README.md):** prediction format, processing stages, and output files.
- **[Inspect or change prompts](experiments/prompting_strategies/README.md):** experimental conditions and structured response requirements.
- **[Check the installation](tests/README.md):** lightweight development checks and their dependencies.

For comparable new runs, retain the sample or split manifest and record the code revision, checkpoint identity, dependency versions, and experiment settings. The repository supports research replication and extension; its automated checks cover software scaffolding rather than replication of the empirical findings.

## Citation

If you build on this work, please cite the paper:

```bibtex
@misc{muki2026llmsconflictmonitoring,
  title         = {Are LLMs Ready for Conflict Monitoring? Empirical Evidence from West Africa},
  author        = {Muki, Hoffmann and Owolabi, Olukunle},
  year          = {2026},
  eprint        = {2605.04177},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2605.04177},
  url           = {https://arxiv.org/abs/2605.04177}
}
```
