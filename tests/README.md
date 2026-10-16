# Checking a development environment

Run the scripts directly from the repository root with the project environment activated:

```bash
PYTHONPATH=. python tests/test_generic_pipeline.py
PYTHONPATH=. python tests/test_counterfactual.py
```

The first command exercises country/path setup and zero-shot prompt construction. The second loads a Nigeria sample and checks counterfactual analyzer construction and perturbation generation. It needs the counterfactual imports, including Captum, and `datasets/nga/state_actor_sample_nga_1000.csv` or another matching sample; retrieve dataset contents with `git lfs pull` if necessary. Neither script performs model inference.

Read the terminal output and exit status. Both scripts signal failure by returning a boolean and converting it to an exit code in their main block; pytest collection alone can miss a returned `False`. Some printed import checks are placeholders, so a successful run establishes only the operations above.

The counterfactual script creates `test_counterfactual_output/` and can initialize a results directory. Its printed `tools/...` examples refer to older paths; current commands are in the [experiment guide](../experiments/README.md). These development checks do not measure model performance or reproduce the paper's statistical results.
