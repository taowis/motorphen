#!/usr/bin/env bash
# Create the project's own Python environment on a CUDA workstation (run there,
# from the project directory). Uses uv; PyTorch from the CUDA 12.8 wheel index.
set -euo pipefail
cd "$(dirname "$0")/../.."
UV=$(command -v uv || echo "$HOME/.local/bin/uv")
"$UV" venv --python 3.12 .venv
"$UV" pip install --python .venv/bin/python -r requirements-gpu.txt \
  --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match
"$UV" pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
"$UV" pip freeze --python .venv/bin/python > results/env_freeze.txt
