#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${project_dir}/.venv"

python3 -m venv "${venv_dir}"
"${venv_dir}/bin/python" -m pip install --upgrade pip setuptools wheel
"${venv_dir}/bin/python" -m pip install -e "${project_dir}[all]"
"${venv_dir}/bin/python" -c "import hexaform; print('hexaform', hexaform.__version__)"
"${venv_dir}/bin/hexaform" --help >/dev/null

echo "Setup complete. Activate with: source .venv/bin/activate"
