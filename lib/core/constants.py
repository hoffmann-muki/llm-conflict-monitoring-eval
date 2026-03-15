"""Project-wide constants for labels and event classes."""
LABEL_MAP = {
    "Violence against civilians": "V",
    "Battles": "B",
    "Explosions/Remote violence": "E",
    "Protests": "P",
    "Riots": "R",
    "Strategic developments": "S"
}

EVENT_CLASSES_FULL = [
    "Violence against civilians",
    "Battles",
    "Explosions/Remote violence",
    "Protests",
    "Riots",
    "Strategic developments"
]

# Source CSV used by country pipelines
CSV_SRC = "datasets/Africa_lagged_data_up_to-2024-10-24.csv"

WORKING_MODELS = [
    "llama3.2:3b",
    "mistral:7b",
    "gemma3:4b",
    "olmo2:7b",
    "acled-small-llm-ft:latest",
]

# Mapping of Ollama model names to local HuggingFace model paths.
# Used for fine-tuning small LLMs when HF_HUB_OFFLINE=1.
# Maps Ollama model IDs (e.g., "llama3.2:3b") to local filesystem paths.
LOCAL_BASE_MODELS = {
    "llama3.2:3b": "models/Llama-3.2-3B",
    "mistral:7b": "models/Mistral-7B-v0.3",
    "gemma3:4b": "models/gemma-3-4b-pt",
    "olmo2:7b": "models/OLMo-2-1124-7B",
    "acled-small-llm-ft:latest": "models/small_llm_merged_acled_v1_seed42",
}

# Country name mapping used across pipelines
COUNTRY_NAMES = {
    'cmr': 'Cameroon',
    'nga': 'Nigeria',
}
