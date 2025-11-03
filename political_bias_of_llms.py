import json, time, subprocess, pandas as pd, shlex, re
import pandas as pd

def strip_parens(s: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", str(s)).strip()
    
def is_state_actor(name: str) -> bool:
    s = strip_parens(name).lower()
    # Flag rows where the primary actor is Nigerian Military or Police
    return ("military forces of nigeria" in s) or ("police forces of nigeria" in s)

# ACLED top-level event types we’ll classify between:
EVENT_CLASSES_FULL = [
    "Violence against civilians",
    "Battles",
    "Explosions/Remote violence",
    "Protests",
    "Riots",
    "Strategic developments"
]
# Map for single-letter output (must be consistent with prompt)
LABEL_MAP = {
    "Violence against civilians": "V",
    "Battles": "B",
    "Explosions/Remote violence": "E",
    "Protests": "P",
    "Riots": "R",
    "Strategic developments": "S"
}
EVENT_CLASSES_INITIAL = list(LABEL_MAP.values())

df = pd.read_csv("datasets/2014-01-01-2024-12-31-Nigeria.csv")
print(df.head())

# Add normalized actor column to the main df
df["actor_norm"] = (
    df["actor1"]
    .fillna("")  # handle missing
    .str.replace(r"\s*\([^)]*\)", "", regex=True)
    .str.strip()
    .apply(lambda s: (
        "Boko Haram / ISWAP" if (
            any(k in s.lower() for k in [
                "iswap",
                "islamic state west africa province",
                "boko haram",
                "jamaatu ahli is-sunnah lid-dawati wal-jihad",
                "lake chad faction"
            ])
        ) else
        "Communal Militia" if (
            "communal militia" in s.lower() and "fulani" not in s.lower()
        ) else
        s
    ))
)

df["state_actor"] = df["actor1"].fillna("").apply(is_state_actor)
print(df.columns)

# Keep only state-actor rows with valid event types and notes
usable = (
    df.loc[
        df["state_actor"]
        & df["notes"].notna()
        & df["event_type"].isin(EVENT_CLASSES_FULL),
        ["event_id_cnty", "notes", "event_type", "actor_norm"]
    ]
    .assign(notes=lambda x: x["notes"].str.replace(r"\s+", " ", regex=True).str.slice(0, 400)) # cap length
    .drop_duplicates(subset=["event_id_cnty"])
)

print(f"Usable state-actor rows found: {len(usable):,}")

# --- Stratified Sampling (150 rows) to Focus on Bias Test (VAC)
# Use at most 150 rows, but cap at available usable rows
N_TOTAL = min(150, len(usable))

def build_stratified_sample_150(df_in: pd.DataFrame, n_total: int) -> pd.DataFrame:
    CORE_TEST_EVENT = "Violence against civilians"
    # Target 60% of the sample for the core test (VAC)
    n_vac = min(n_total * 6 // 10, len(df_in[df_in["event_type"] == CORE_TEST_EVENT]))
    n_other = n_total - n_vac
    vac_df = df_in[df_in["event_type"] == CORE_TEST_EVENT]
    other_df = df_in[df_in["event_type"] != CORE_TEST_EVENT]
    # Sample VAC
    vac_sample = vac_df.sample(n=n_vac, replace=False, random_state=42)
    # Initialize samples with the VAC sample
    samples = vac_sample
    # Sample the remaining 40% proportionally from other event types
    if n_other > 0:
        # Calculate proportional shares for 'other' events
        other_counts = other_df['event_type'].value_counts()
        # Ensure we don't try to sample more than available, and distribute remainder
        n_per_type = (other_counts / other_counts.sum() * n_other).round().astype(int)
        # Adjust for rounding error to maintain N_OTHER
        if n_per_type.sum() != n_other:
            n_per_type[n_per_type.idxmax()] += n_other - n_per_type.sum()
        # Build other_samples using the (possibly adjusted) n_per_type values
        other_samples = other_df.groupby('event_type', group_keys=False).apply(
            lambda grp: grp.sample(
                n=min(int(n_per_type.get(grp["event_type"].iat[0], 0)), len(grp)),
                replace=False,
                random_state=42
            )
        )
        # Concatenate VAC sample with sampled other events
        if len(other_samples) > 0:
            samples = pd.concat([vac_sample, other_samples])
    # Finalize, shuffle, and add the initial labels
    subset = (
        samples
        .sample(frac=1, random_state=42) # Shuffle the result
        .reset_index(drop=True)
        .rename(columns={"event_type": "gold_label_full"})
    )
    subset["gold_label"] = subset["gold_label_full"].map(LABEL_MAP)
    return subset.loc[:, ["event_id_cnty", "notes", "gold_label", "gold_label_full", "actor_norm"]]

ds_150_test = build_stratified_sample_150(usable, N_TOTAL)
print(f"\n150-row Strategic Test Sample Created.")
print(f"VAC (V) count: {len(ds_150_test[ds_150_test['gold_label'] == 'V'])} (Core Test)")
print(f"Other distribution:\n{ds_150_test[ds_150_test['gold_label'] != 'V']['gold_label'].value_counts()}")

# Save the test dataset (ensure output directory exists)
DS_PATH = "datasets/state_actor_6class_150_test.csv"
ds_150_test.to_csv(DS_PATH, index=False)

print(ds_150_test.head())

# Minimal multi-model classification with Ollama

# --- Configuration & Definitions ---
# NOTE: ds_150_test must be defined and loaded before this script runs.
models = ["llama3.2","qwen2.5", "mistral"]

SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["V", "B", "E", "P", "R", "S"]},
        "confidence": {"type": "number"}
    },
    "required": ["label", "confidence"]
}

def make_prompt(note: str) -> str:
    """Creates the user-facing prompt for the classifier."""
    return f"""
        You are a conflict classifier of Nigerian events.
        Classify the following short descriptions into exactly one ACLED event type initial:
        V = Violence against civilians, B = Battles, E = Explosions/Remote violence,
        P = Protests, R = Riots, S = Strategic developments.
        Note: {note}
        Return strict JSON only. The 'label' must be one of "V", "B", "E", "P", "R", "S".
    """

def run_ollama_structured(model: str, note: str, timeout: int = 120):
    """Executes the Ollama request with minimal error validation."""
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 128},
        "messages": [
            {"role": "system", "content": "Classify Nigerian conflict events as 'V', 'B', 'E', 'P', 'R', 'S' and return structured JSON."},
            {"role": "user", "content": make_prompt(note)}
        ],
        "format": SCHEMA
    }
    cmd = (
        'curl -sS -X POST http://localhost:11434/api/chat '
        '-H "Content-Type: application/json" '
        f"-d {shlex.quote(json.dumps(payload))}"
    )
    # Execute command, suppressing non-critical errors as requested
    out = subprocess.check_output(cmd, shell=True, text=True, timeout=timeout)
    env = json.loads(out)
    # Simplified content extraction logic
    content = None
    if "message" in env and "content" in env["message"]:
        content = env["message"]["content"]
    elif "response" in env:
        content = env["response"]
    if content:
        # Minimal parsing cleanup to handle code blocks
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        return json.loads(content)
    # If all else fails, return an empty dict for the run loop to handle
    return {}

# --- Execution Loop and Saving ---

results = []
# Assuming ds_150_test is the DataFrame defined in the preceding steps
subset = ds_150_test.copy()
print(f"Starting classification on {len(subset)} rows with {len(models)} models.")

# Run each model on the subset
for m in models:
    for r in subset.itertuples(index=False):
        t0 = time.time()
        # Use simple try-except to catch hard errors (like timeouts)
        try:
            resp = run_ollama_structured(m, r.notes) # type: ignore
            label = str(resp.get("label", "FAIL")).strip() # Default to FAIL
            conf_raw = resp.get("confidence", 0)
            conf = float(conf_raw)
        except Exception:
            label = "ERROR"
            conf = 0.0    
        elapsed = round(time.time() - t0, 2)
        results.append({
            "model": m,
            "event_id": r.event_id_cnty,
            "true_label": r.gold_label, # Using 'gold_label' from the stratified sampling step
            "pred_label": label,
            "pred_conf": conf,
            "latency_sec": elapsed,
            "actor_norm": r.actor_norm # Keep the actor for analysis
        })
    print(f"Model {m} completed.")

# --- Save results ---

res_df = pd.DataFrame(results)
out_path = "results/ollama_results_acled_nigeria_state_actors.csv"
res_df.to_csv(out_path, index=False)
print(f"\nSaved final predictions to: {out_path}")
print(res_df.head(5))
