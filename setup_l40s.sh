#!/usr/bin/env bash
# One-time setup on the L40S box. Usage:
#   ./setup_l40s.sh [cuXXX] [--prefetch]
# cuXXX defaults to cu128 - check `nvidia-smi` (top-right "CUDA Version") and use
# the closest matching wheel index (cu128 / cu124 / cu121).
# --prefetch downloads all models (~4GB) now instead of on first request.
set -euo pipefail
cd "$(dirname "$0")"

CUDA_TAG="cu128"
PREFETCH=0
for arg in "$@"; do
  case "$arg" in
    --prefetch) PREFETCH=1 ;;
    cu*) CUDA_TAG="$arg" ;;
    *) echo "unknown arg: $arg" && exit 1 ;;
  esac
done

echo "== GPU =="
nvidia-smi | head -n 10 || { echo "nvidia-smi failed - is this the GPU box?"; exit 1; }

echo "== venv =="
# clean, isolated venv - never --system-site-packages (causes CPU fallback / libcupti issues)
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel

echo "== torch (${CUDA_TAG}) =="
# torch MUST be installed from the CUDA index BEFORE whisperx, otherwise pip
# pulls the CPU-only build from PyPI.
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

python - <<'PY'
import torch
assert torch.cuda.is_available(), (
    "torch sees no CUDA. Wrong cuXXX tag for this driver, or venv is polluted. "
    "Delete .venv and re-run with the tag matching `nvidia-smi` (e.g. ./setup_l40s.sh cu124)."
)
print(f"torch {torch.__version__} cuda {torch.version.cuda} - {torch.cuda.get_device_name(0)}")
PY

echo "== python deps =="
pip install -r requirements.txt

echo "== ffmpeg =="
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found - attempting apt install (needed for mp3/audio decode)"
  sudo apt-get update -y && sudo apt-get install -y ffmpeg \
    || echo "WARNING: could not install ffmpeg (no sudo?). wav uploads may still work; mp3 will not."
fi

if [[ "$PREFETCH" == "1" ]]; then
  echo "== prefetching models (~4GB) =="
  python - <<'PY'
import os
import torch  # noqa: F401  (load torch's cuDNN first)
import whisperx
model_name = os.environ.get("WHISPER_MODEL", "large-v3")
print(f"downloading whisper model: {model_name}")
whisperx.load_model(model_name, "cuda", compute_type=os.environ.get("COMPUTE_TYPE", "float16"))
for lang in ("en", "hi"):
    print(f"downloading align model: {lang}")
    whisperx.load_align_model(language_code=lang, device="cuda")
print("prefetch done")
PY
fi

echo
echo "Setup complete. Start the server with:  tmux new -s whisper_api ./run.sh"
