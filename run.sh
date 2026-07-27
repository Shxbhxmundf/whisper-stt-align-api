#!/usr/bin/env bash
# Start the API (run inside tmux: tmux new -s whisper_api ./run.sh)
# Env: PORT (8888), WHISPER_MODEL (large-v3), COMPUTE_TYPE (float16), BATCH_SIZE (8)
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

# Make sure ctranslate2 can find the pip-bundled NVIDIA libs (cuDNN etc.) if it
# dlopens them - belt-and-suspenders against the classic libcudnn load error.
NVIDIA_LIBS="$(python - <<'PY'
import glob, os, sysconfig
sp = sysconfig.get_paths()["purelib"]
print(":".join(sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))))
PY
)"
if [[ -n "$NVIDIA_LIBS" ]]; then
  export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8888}"
