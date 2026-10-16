#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
env_path="${1:-.venv}"
python_bin="${PYTHON:-python3.12}"
if [[ -e "$env_path" ]]; then
  echo "Environment already exists: $env_path; activate it or choose another path." >&2
  exit 1
fi
"$python_bin" -m venv --prompt sdapt "$env_path"
"$env_path/bin/python" -m pip install --upgrade pip
"$env_path/bin/python" -m pip install -r requirements.txt
"$env_path/bin/python" -m pip check
printf 'Activate with: source "%s/bin/activate"\n' "$env_path"
