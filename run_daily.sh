#!/bin/zsh
# Daily forecast update: fetch every source, re-run the model, rebuild the site.
# Usage: ./run_daily.sh            (full run, ~10-15 minutes because Wikipedia rate-limits)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
LOG="logs/run_$(date +%Y-%m-%d).log"
if [ ! -x .venv/bin/python ]; then
  uv venv .venv -q && uv pip install -q --python .venv/bin/python -r requirements.txt
fi
.venv/bin/python -m pipeline.run "$@" 2>&1 | tee "$LOG"
echo "Site rebuilt: $(pwd)/site/index.html"
