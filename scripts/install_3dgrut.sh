#!/usr/bin/env bash
# Ставит 3DGRUT (NVIDIA, Gaussian splatting; 3DGUT учится прямо на fisheye) в
# third_party/3dgrut/.venv: свой Python 3.11, локальная CUDA 12.8 (RTX 50 / Blackwell),
# PyTorch 2.8 — без sudo и без системной CUDA. ~10 ГБ, 20–40 мин.
#   scripts/install_3dgrut.sh  > out/install_3dgrut.log 2>&1
# После: source third_party/3dgrut/.venv/bin/activate (CUDA_HOME — см. ниже).
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT/third_party/3dgrut"
BIN=$ROOT/third_party/bin
mkdir -p "$BIN"
if [ ! -x "$BIN/uv" ]; then  # uv берём из pip нашего .venv — без curl | sh
    "$ROOT/.venv/bin/pip" install -q uv
    ln -sf "$ROOT/.venv/bin/uv" "$BIN/uv"
fi
export PATH=$BIN:$PATH
# Python 3.11 и кэш uv — в проекте: из терминала VS Code (snap) HOME подменён на
# ~/snap/code/<ревизия>, и окружение сломалось бы после обновления VS Code.
export UV_PYTHON_INSTALL_DIR=$ROOT/third_party/python UV_CACHE_DIR=$ROOT/third_party/.uv-cache
# Установщик CUDA (makeself) при DISPLAY и без терминала пытается открыть xterm и падает
# ("exec: -title: not found") — ставим без окна.
unset DISPLAY
rm -rf .venv
FORCE_LOCAL_CUDA=1 CUDA_VERSION=12.8 ./scripts/create_venv_cuda.sh 3dgrut
# create_venv_cuda.sh ставит CUDA в .venv/cuda-*, но CUDA_HOME дальше не передаёт.
CUDA_HOME=$(ls -d "$PWD"/.venv/cuda-12.8* | head -1)
export CUDA_HOME PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
source .venv/bin/activate
./install_env_uv.sh 3dgrut
# Установочный файл CUDA (~4 ГБ) и кэш колёс больше не нужны.
rm -f /tmp/cuda_12.8*_linux.run
uv cache clean
python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "[3dgrut] готово; CUDA_HOME=$CUDA_HOME"
