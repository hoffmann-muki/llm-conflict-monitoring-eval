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

WORKING_MODELS = ["llama3.2", "qwen3:8b", "mistral:7b", "gemma3:7b", "olmo2:7b"]
