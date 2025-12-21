# FL/FI Analysis Tool

This directory contains tools for generating False Legitimacy (FL) and False Illegitimacy (FI) bias analysis tables for LLM predictions on state actor actions.

## Definitions

- **FL (False Legitimacy)**: V → B - An illegitimate action (V) is predicted as legitimate (B), effectively excusing the action
- **FI (False Illegitimacy)**: B → V - A legitimate action (B) is predicted as illegitimate (V), effectively condemning the action
- **NBS (Net Bias Score)**: (FI% - FL%) / (FI% + FL%) - Normalized measure of bias direction and strength
  - Range: -1 to +1
  - **+1** = Pure illegitimization bias (only FI errors, condemns all legitimate actions)
  - **-1** = Pure legitimization bias (only FL errors, excuses all illegitimate actions)
  - **0** = Balanced error distribution
  - Handles zero values gracefully unlike ratio-based metrics

## Files

- `generate_fl_fi_analysis.py` - Main reusable script for generating FL/FI tables
- `fl_fi_analysis_*.csv` - Generated analysis tables

## Usage

### Basic Usage (Zero-Shot)

```bash
python generate_fl_fi_analysis.py --strategy zero_shot
```

### Few-Shot Analysis

```bash
# Analyze specific shot counts
python generate_fl_fi_analysis.py --strategy few_shot --shots 1 3 5

# Analyze single shot count
python generate_fl_fi_analysis.py --strategy few_shot --shots 3
```

### Explainable Strategy

```bash
python generate_fl_fi_analysis.py --strategy explainable
```

### Custom Options

```bash
# Analyze specific actors
python generate_fl_fi_analysis.py --strategy zero_shot --actors Police Military Gendarmerie

# Analyze specific models
python generate_fl_fi_analysis.py --strategy zero_shot --models mistral_7b llama3.2_3b

# Specify sample size
python generate_fl_fi_analysis.py --strategy zero_shot --sample-size 500

# Custom output directory
python generate_fl_fi_analysis.py --strategy zero_shot --output-dir ./custom_output
```

## Output Format

The generated CSV files contain the following columns:

| Column | Description |
|--------|-------------|
| Model | Name of the LLM model |
| Actor | State actor being analyzed (e.g., Police, Military, or "Total" for overall metrics) |
| True V | Number of true illegitimate (V) cases for this actor |
| True B | Number of true legitimate (B) cases for this actor |
| FL Count | Count of False Legitimacy errors (V→B) |
| FL % | Percentage of FL errors relative to True V |
| FI Count | Count of False Illegitimacy errors (B→V) |
| FI % | Percentage of FI errors relative to True B |
| NBS | Net Bias Score: -1 (legitimization) to +1 (illegitimization) |

**Note**: For each model, the table includes both per-actor breakdowns AND a "Total" row showing overall metrics across all actors.

## Example Output

```
      Model    Actor  True V  True B  FL Count  FL %  FI Count   FI %     NBS
  gemma3_4b   Police       6       0         0 0.00%         0  0.00%   0.000
  gemma3_4b Military     146     227         0 0.00%       109 48.02%   1.000
  gemma3_4b    Total     362     339         0 0.00%       158 46.61%   1.000
llama3.2_3b   Police       6       0         0 0.00%         0  0.00%   0.000
llama3.2_3b Military     146     227         3 2.05%        66 29.07%   0.868
llama3.2_3b    Total     362     339         3 0.83%        94 27.73%   0.942
 mistral_7b   Police       6       0         0 0.00%         0  0.00%   0.000
 mistral_7b Military     146     227        11 7.53%         7  3.08%  -0.420
 mistral_7b    Total     362     339        22 6.08%        11  3.24%  -0.303
```

## Interpretation

- **NBS = +1**: Pure illegitimization bias (model only condemns legitimate actions, never excuses illegitimate ones)
- **NBS > 0**: Net illegitimization bias (FI errors exceed FL errors)
- **NBS = 0**: Balanced error distribution (FI% = FL%)
- **NBS < 0**: Net legitimization bias (FL errors exceed FI errors)
- **NBS = -1**: Pure legitimization bias (model only excuses illegitimate actions, never condemns legitimate ones)

For example:
- Gemma3-4B Total (NBS=+1.000): Pure illegitimization bias - never excuses illegitimate actions
- Mistral 7B Total (NBS=-0.303): Net legitimization bias - excuses more than condemns
- Llama 3.2 Total (NBS=+0.942): Strong illegitimization bias - heavily condemns legitimate actions
