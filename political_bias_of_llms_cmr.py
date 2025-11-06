import pandas as pd
import json, time, re, os, subprocess, shlex
from country_data_exploration import extract_country_rows
from column_data_extraction import get_actor_norm_series, extract_state_actor
from input_data_extraction import build_stratified_sample

# Map for single-letter output (must be consistent with prompt)
LABEL_MAP = {
    "Violence against civilians": "V",
    "Battles": "B",
    "Explosions/Remote violence": "E",
    "Protests": "P",
    "Riots": "R",
    "Strategic developments": "S"
}

# ACLED top-level event types we’ll classify between:
EVENT_CLASSES_FULL = [
    "Violence against civilians",
    "Battles",
    "Explosions/Remote violence",
    "Protests",
    "Riots",
    "Strategic developments"
]

CSV_SRC = "datasets/Africa_lagged_data_up_to-2024-10-24.csv"
OUT_CAM = "datasets/Cameroon_lagged_data_up_to-2024-10-24.csv"

if not os.path.exists(CSV_SRC):
    raise SystemExit(f"Source CSV not found: {CSV_SRC}")

df_all = pd.read_csv(CSV_SRC)
df_cam = extract_country_rows(CSV_SRC, "Cameroon")

# Resolve column names case-insensitively to match ACLED casing differences
cols_lower = {c.lower(): c for c in df_cam.columns}
col_actor = cols_lower.get('actor1', 'actor1')
col_notes = cols_lower.get('notes', 'notes')
col_event_type = cols_lower.get('event_type', 'event_type')
col_event_id = cols_lower.get('event_id_cnty', 'event_id_cnty')

# Create normalized actor column using the configurable helper
df_cam["actor_norm"] = get_actor_norm_series(df_cam, actor_col=col_actor)

# Create state_actor boolean using helper (country-specific)
df_cam["state_actor"] = extract_state_actor(df_cam, country="cameroon", actor_col=col_actor)

# Keep only state-actor rows with valid event types and notes
usable = (
    df_cam.loc[
        df_cam["state_actor"]
            & df_cam[col_notes].notna()
            & df_cam[col_event_type].isin(EVENT_CLASSES_FULL),
            [col_event_id, col_notes, col_event_type, "actor_norm"]
    ]
    .rename(columns={col_event_id: "event_id_cnty", col_notes: "notes", col_event_type: "event_type"})
    .assign(notes=lambda x: x["notes"].str.replace(r"\s+", " ", regex=True).str.slice(0, 400))
    .drop_duplicates(subset=["event_id_cnty"])
)

print(f"Usable state-actor rows found (Cameroon): {len(usable):,}")

# --- Build stratified sample using the generic sampler (default. 100 samples) --- #
SAMPLE_SIZE = int(os.environ.get('SAMPLE_SIZE', '100'))
N_TOTAL = min(SAMPLE_SIZE, len(usable))
df_test = build_stratified_sample(
    usable,
    stratify_col='event_type',
    n_total=N_TOTAL,
    primary_group='Violence against civilians',
    primary_share=0.6,
    label_map=LABEL_MAP,
    random_state=42,
    replace=False
)

SAMPLE_PATH = "datasets/state_actor_100_sample_cmr.csv"
df_test.to_csv(SAMPLE_PATH, index=False)
print(f"Wrote stratified sample to {SAMPLE_PATH}")
print(df_test.head())

# Minimal multi-model classification with Ollama

models = ["llama3.2", "qwen2.5", "mistral"]

SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["V", "B", "E", "P", "R", "S"]},
    "confidence": {"type": "number"},
    # optional per-class numeric scores in the order [V,B,E,P,R,S]
    "logits": {"type": "array", "items": {"type": "number"}}
    },
    "required": ["label", "confidence"]
}

def make_prompt(note: str) -> str:
    return f"""
        You are a conflict classifier of Cameroonian events.
        Classify the following short descriptions into exactly one ACLED event type initial:
        V = Violence against civilians, B = Battles, E = Explosions/Remote violence,
        P = Protests, R = Riots, S = Strategic developments.
        Note: {note}
    Return strict JSON only. The 'label' must be one of "V", "B", "E", "P", "R", "S".
    Additionally return a numeric array field named "logits" with six scores in the exact order [V,B,E,P,R,S] representing unnormalized model scores (higher means more likely). This allows downstream per-class probability calibration.
    """

def run_ollama_structured(model: str, note: str, timeout: int = 120):
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 128},
        "messages": [
            {"role": "system", "content": "Classify Cameroonian conflict events as 'V','B','E','P','R','S' and return structured JSON."},
            {"role": "user", "content": make_prompt(note)}
        ],
        "format": SCHEMA
    }
    cmd = (
        'curl -sS -X POST http://localhost:11434/api/chat '
        '-H "Content-Type: application/json" '
        f"-d {shlex.quote(json.dumps(payload))}"
    )
    out = subprocess.check_output(cmd, shell=True, text=True, timeout=timeout)
    env = json.loads(out)
    content = None
    if "message" in env and "content" in env["message"]:
        content = env["message"]["content"]
    elif "response" in env:
        content = env["response"]
    if content:
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        return json.loads(content)
    return {}

results = []
subset = df_test.copy()
print(f"Starting classification on {len(subset)} rows with {len(models)} models (serial execution).")

def run_model_on_rows(model_name, rows):
    out = []
    for r in rows.itertuples(index=False):
        t0 = time.time()
        try:
            resp = run_ollama_structured(model_name, r.notes)
            label = str(resp.get("label", "FAIL")).strip()
            conf = float(resp.get("confidence", 0))
            # Capture any per-class scores/logits if the model provides them.
            # Common keys: 'logits', 'log_probs', 'scores', 'label_scores'
            logits = None
            for k in ("logits", "log_probs", "scores", "label_scores"):
                if k in resp:
                    logits = resp.get(k)
                    break
        except Exception:
            label = "ERROR"
            conf = 0.0
            logits = None
        elapsed = round(time.time() - t0, 2)
        out.append({
            "model": model_name,
            "event_id": r.event_id_cnty,
            "true_label": r.gold_label,
            "pred_label": label,
            "pred_conf": conf,
            "logits": json.dumps(logits) if logits is not None else None,
            "latency_sec": elapsed,
            "actor_norm": r.actor_norm
        })
    return out

for m in models:
    print(f"Starting model: {m}")
    model_results = run_model_on_rows(m, subset)
    results.extend(model_results)
    print(f"Model {m} completed.")

res_df = pd.DataFrame(results)
out_path = "results/ollama_results_acled_cameroon_state_actors.csv"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
res_df.to_csv(out_path, index=False)
print(f"\nSaved final predictions to: {out_path}")
print(res_df.head(5))
