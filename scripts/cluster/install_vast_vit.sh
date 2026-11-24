#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
base_python=/venv/main/bin/python
env_dir="$repo_root/.cache/vit-venv"

if [[ ! -x "$base_python" ]] || ! command -v uv >/dev/null; then
  echo 'This installer needs the Vast.ai PyTorch image with /venv/main and uv.' >&2
  exit 1
fi

if [[ ! -x "$env_dir/bin/python" ]]; then
  uv venv --python "$base_python" --system-site-packages "$env_dir"
fi

# Inherit Vast's torch and torchvision and install the ViT dependencies at the repo's pins.
uv pip install --python "$env_dir/bin/python" --cache-dir "$repo_root/.cache/uv" \
  --constraint "$repo_root/requirements.txt" \
  transformers tokenizers safetensors huggingface-hub numpy pillow pyarrow \
  pyyaml scipy scikit-learn tqdm

"$env_dir/bin/python" - <<'PY'
import pyarrow.parquet
import sklearn
import torch
import torchvision
import yaml
from transformers import AutoImageProcessor, AutoModelForImageClassification

torch.zeros(1, device="cuda")
print(f"ViT environment ready on {torch.cuda.get_device_name()}")
print(f"PyTorch {torch.__version__}, torchvision {torchvision.__version__}, CUDA {torch.version.cuda}")
PY

printf 'To activate the ViT environment, run this command.\n  source %q\n' \
  "$env_dir/bin/activate"
