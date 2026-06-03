#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

exec conda run -n agent-solution-langgraph langgraph dev --config langgraph.json --no-browser "$@"

