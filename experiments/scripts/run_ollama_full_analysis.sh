#!/bin/bash
# Compatibility shim.
# Canonical unified pipeline entrypoint is now:
#   experiments/scripts/run_full_analysis.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$SCRIPT_DIR/run_full_analysis.sh"

echo "[DEPRECATED] run_ollama_full_analysis.sh now forwards to run_full_analysis.sh" >&2
exec "$TARGET" "$@"
