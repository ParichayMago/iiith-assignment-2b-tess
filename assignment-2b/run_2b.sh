#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo 'Usage: bash run_2b.sh /path/to/tess.zip' >&2
  exit 2
fi
project=$(cd -- "$(dirname -- "$0")" && pwd)
archive=$(cd -- "$(dirname -- "$1")" && pwd)/$(basename -- "$1")
cd "$project"
python=${REVIEW_PYTHON:-"$project/.venv/bin/python"}
mkdir -p .local/logs
"$python" scripts/audit_dataset.py "$archive"
"$python" scripts/prepare_features.py
"$python" -m pytest -q tests
"$python" scripts/run_experiments.py --stage train
"$python" scripts/run_experiments.py --stage test
"$python" scripts/analyze_results.py
"$python" scripts/analyze_acoustics.py
"$python" scripts/build_report.py
"$python" scripts/verify_results.py
