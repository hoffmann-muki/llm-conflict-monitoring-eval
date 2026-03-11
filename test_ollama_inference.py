#!/usr/bin/env python3
"""Quick test of ollama inference with acled-small-llm-ft model."""
import sys
import os
sys.path.insert(0, os.getcwd())

from experiments.prompting_strategies import ZeroShotStrategy
from lib.inference.ollama_client import run_ollama_structured

# Test event from ACLED
test_note = "On 11 June 2023, an unidentified armed group attacked a civilian settlement, killing 5 people and injuring 3 others."

print("Testing ollama inference with acled-small-llm-ft")
print("=" * 70)
print(f"Test note: {test_note}\n")

strategy = ZeroShotStrategy()
prompt = strategy.make_prompt(test_note)
system_msg = strategy.get_system_message()

print(f"System message:\n{system_msg}\n")
print(f"Prompt:\n{prompt}\n")
print("=" * 70)
print("Running inference...\n")

try:
    result = run_ollama_structured(
        "acled-small-llm-ft",
        prompt,
        system_msg,
        schema=strategy.get_schema(),
        timeout=120
    )
    
    print("✓ Inference successful!")
    print(f"Response:\n{result}\n")
    print("=" * 70)
    print(f"Predicted label: {result.get('label')}")
    print(f"Confidence: {result.get('confidence')}")
    if result.get('logits'):
        print(f"Logits: {result.get('logits')}")
    
except Exception as e:
    print(f"✗ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
