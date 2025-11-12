import json, time, re, subprocess, shlex
from typing import Iterable, List

# Shared JSON schema for structured classification responses
SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "confidence": {"type": "number"},
        "logits": {"type": "array", "items": {"type": "number"}}
    },
    "required": ["label", "confidence"]
}

def make_prompt(note: str) -> str:
     return f"""Classify this event: {note}

Categories: V=Violence against civilians, B=Battles, E=Explosions, P=Protests, R=Riots, S=Strategic developments

Answer with JSON only: {{"label": "V", "confidence": 0.9, "logits": [0.9,0.1,0.0,0.0,0.0,0.0]}}"""

def run_ollama_structured(model: str, note: str, system_msg: str | None = None, schema=None, timeout: int = 120):
    """Run a single structured request against local Ollama and return parsed JSON.

    Parameters
    - model: model name (e.g., 'gemma:7b')
    - note: short text to classify
    - system_msg: optional system message to include for context
    - schema: optional JSON schema dict; defaults to SCHEMA
    - timeout: seconds for the curl call
    """
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": 0.0},
        "messages": [
            {"role": "user", "content": make_prompt(note)}
        ]
    }
    cmd = (
        'curl -sS -X POST http://localhost:11434/api/chat '
        '-H "Content-Type: application/json" '
        f"-d {shlex.quote(json.dumps(payload))}"
    )
    out = subprocess.check_output(cmd, shell=True, text=True, timeout=timeout)
    env = json.loads(out)
    
    # Try to find JSON content in the response
    raw_text = None
    if isinstance(env, dict):
        msg = env.get('message') if isinstance(env.get('message'), dict) else None
        if msg:
            # Check content first, then thinking as fallback
            if msg.get('content'):
                raw_text = msg.get('content')
            elif msg.get('thinking'):
                raw_text = msg.get('thinking')
        elif env.get('response'):
            raw_text = env.get('response')
    
    if raw_text:
        # Try to parse as direct JSON first
        try:
            return json.loads(raw_text.strip())
        except:
            # If that fails, look for JSON object in the text
            json_match = re.search(r'\{[^{}]*"label"\s*:\s*"[VBEPRS]"[^{}]*\}', raw_text)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except:
                    pass
            
            # Try to handle partial JSON by looking for label at least
            label_match = re.search(r'"label"\s*:\s*"([VBEPRS])"', raw_text)
            conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', raw_text)
            if label_match:
                result = {"label": label_match.group(1)}
                if conf_match:
                    try:
                        result["confidence"] = float(conf_match.group(1))
                    except:
                        result["confidence"] = 0.5
                else:
                    result["confidence"] = 0.5
                return result
            
            # Fallback to any JSON object
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except:
                    pass
    
    # Return empty dict if no valid JSON found
    return {}

def run_model_on_rows(model_name: str, rows, note_col: str = 'notes', event_id_cols: Iterable[str] = ('event_id_cnty', 'event_id'),
                      true_label_cols: Iterable[str] = ('gold_label', 'true_label'), actor_norm_col: str = 'actor_norm') -> List[dict]:
    """Run `model_name` on a pandas rows object (DataFrame) and return list of result dicts.

    The function will try to pull event id and true label from common column names if present.
    """
    out = []
    for r in rows.itertuples(index=False):
        t0 = time.time()
        try:
            note = getattr(r, note_col)
            resp = run_ollama_structured(model_name, note)
            label = str(resp.get("label", "FAIL")).strip()
            conf = float(resp.get("confidence", 0))
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
        # resolve event_id and true_label from possible fields
        event_id = None
        for c in event_id_cols:
            event_id = getattr(r, c, None) or event_id
        true_label = None
        for c in true_label_cols:
            true_label = getattr(r, c, None) or true_label
        out.append({
            "model": model_name,
            "event_id": event_id,
            "true_label": true_label,
            "pred_label": label,
            "pred_conf": conf,
            "logits": json.dumps(logits) if logits is not None else None,
            "latency_sec": elapsed,
            "actor_norm": getattr(r, actor_norm_col, None)
        })
    return out
