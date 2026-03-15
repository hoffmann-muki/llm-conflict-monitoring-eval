from __future__ import annotations

import json
import os
import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _extract_json_object(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start:i + 1]
                try:
                    parsed = json.loads(snippet)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    return None
    return None


def _build_generation_prompt(user_prompt: str, system_msg: str | None) -> str:
    if system_msg:
        return (
            "### System\n"
            f"{system_msg}\n\n"
            "### User\n"
            f"{user_prompt}\n\n"
            "### Assistant\n"
        )
    return (
        "### User\n"
        f"{user_prompt}\n\n"
        "### Assistant\n"
    )


def parse_hf_model_path_map(raw: str | None) -> dict[str, str]:
    model_map: dict[str, str] = {}
    if not raw:
        return model_map
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid HF_MODEL_PATH_MAP entry '{item}'. Expected format 'model_name=/path/to/model'."
            )
        model_name, model_path = item.split("=", 1)
        model_name = model_name.strip()
        model_path = model_path.strip()
        if not model_name or not model_path:
            raise ValueError(
                f"Invalid HF_MODEL_PATH_MAP entry '{item}'. Model name and path are both required."
            )
        model_map[model_name] = model_path
    return model_map


def get_hf_inference_models() -> set[str]:
    raw = os.environ.get("HF_INFERENCE_MODELS", "")
    return {m.strip() for m in raw.split(",") if m.strip()}


def is_hf_inference_model(model_name: str) -> bool:
    return model_name in get_hf_inference_models()


def resolve_hf_model_path(model_name: str) -> str:
    path_map = parse_hf_model_path_map(os.environ.get("HF_MODEL_PATH_MAP"))
    if model_name in path_map:
        return path_map[model_name]

    fallback = os.environ.get("HF_MODEL_PATH", "").strip()
    if fallback:
        return fallback

    raise ValueError(
        f"Model '{model_name}' is configured for HF inference but has no checkpoint path. "
        "Set HF_MODEL_PATH_MAP='model_name=/path/to/checkpoint' or HF_MODEL_PATH=/path/to/checkpoint."
    )


def resolve_hf_device() -> str:
    env_device = os.environ.get("HF_DEVICE", "").strip()
    if env_device:
        return env_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_hf_max_new_tokens(default: int = 96) -> int:
    raw = os.environ.get("HF_MAX_NEW_TOKENS", str(default)).strip()
    try:
        val = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid HF_MAX_NEW_TOKENS value: {raw}") from exc
    if val <= 0:
        raise ValueError(f"HF_MAX_NEW_TOKENS must be > 0, got {val}")
    return val


_RUNTIME_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}


def _load_hf_runtime(model_path: str, device: str):
    cache_key = (model_path, device)
    if cache_key in _RUNTIME_CACHE:
        return _RUNTIME_CACHE[cache_key]

    local_files_only = os.environ.get("HF_HUB_OFFLINE", "") == "1"
    if local_files_only and not os.path.isdir(model_path):
        raise ValueError(
            f"HF_HUB_OFFLINE=1 is set but HF checkpoint path does not exist locally: {model_path}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    )
    model = model.to(device)  # type: ignore[call-arg]
    model.eval()
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        gen_cfg.do_sample = False
        gen_cfg.temperature = 1.0
        gen_cfg.top_p = 1.0

    _RUNTIME_CACHE[cache_key] = (tokenizer, model)
    return tokenizer, model


def run_hf_structured(
    model_name: str,
    prompt: str,
    system_msg: str | None = None,
    schema=None,
    max_new_tokens: int | None = None,
) -> dict:
    """Run local HF causal LM inference and return best-effort structured JSON dict."""
    del schema  # Kept for API parity with Ollama client.

    model_path = resolve_hf_model_path(model_name)
    device = resolve_hf_device()
    if max_new_tokens is None:
        max_new_tokens = resolve_hf_max_new_tokens(default=96)
    elif max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be > 0, got {max_new_tokens}")

    tokenizer, model = _load_hf_runtime(model_path, device)

    generation_prompt = _build_generation_prompt(prompt, system_msg)
    inputs = tokenizer(generation_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_tokens = generated[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    parsed = _extract_json_object(raw_output)
    if parsed is not None:
        return parsed

    # Best-effort fallback to label/confidence extraction.
    label_match = re.search(r'"label"\s*:\s*"([^"\s]+)"', raw_output)
    conf_match = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', raw_output)

    out: dict[str, Any] = {}
    if label_match:
        out["label"] = label_match.group(1)
    if conf_match:
        out["confidence"] = float(conf_match.group(1))
    return out
