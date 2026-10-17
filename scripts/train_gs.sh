#!/usr/bin/env bash
# Gaussian splatting по сессии: 3DGUT (учится прямо на fisheye), маски сканера.
# Стратегия по умолчанию — обычная 3DGS (apps/colmap_3dgut.yaml), как у авторов 3DGUT для
# ScanNet++ (fisheye в помещении). MCMC (CONFIG=apps/colmap_3dgut_mcmc.yaml) в комнате вредна:
# шум позиций уносит невидимые гауссианы за стены (на 7000 шагах в комнате было 4% из 1 млн).
# Набор — scripts/session_to_colmap.py; позы из лидара, поэтому сплат в метрах и совпадает
# с мешем-коллайдером (нормализацию сцены при экспорте выключаем).
#   scripts/train_gs.sh tilt30gs          # позы RKO-LIO: data/sessions/<s>/colmap
#   scripts/train_gs.sh tilt30gs gt       # эталонные позы: colmap_gt — для сравнения
#   EXTRA="n_iterations=7000" scripts/train_gs.sh tilt30gs   # другие параметры 3DGRUT
# Наши настройки (проверены на sklad03): порог уплотнения 5e-5 вместо 2e-4 — сцена в метрах, иначе
# гауссианы почти не делятся и растягиваются; штрафы нашей правки 3DGRUT (scripts/patches/) за
# «иголки» (длинная ось > 10 средних) и за размер больше 0.5 м — это и были длинные линии.
# Итог: out/gs/<s>_<позы>/gaussians.usdz (NuRec, Isaac Sim 5.0+), gaussians.ply, метрики на
# отложенных кадрах (каждый 8-й) — в логе обучения и в папке прогона.
set -e
S=${1:?укажите сессию}
POSES=${2:-rko}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
G=$ROOT/third_party/3dgrut
DATA=$ROOT/data/sessions/$S/colmap
[ "$POSES" = gt ] && DATA=${DATA}_gt
OUT=$ROOT/out/gs/${S}_$POSES
[ -f "$DATA/sparse/0/images.txt" ] || { echo "нет набора $DATA — scripts/session_to_colmap.py $S"; exit 1; }
rm -rf "$OUT/runs"  # прежние прогоны — по 1–2 ГБ, и eval_views мог взять не тот
mkdir -p "$OUT"
CUDA_HOME=$(ls -d "$G"/.venv/cuda-12.8* | head -1)
export CUDA_HOME PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
# Собранные CUDA-модули — в проекте, а не в ~/.cache (из терминала VS Code-snap HOME другой).
export TORCH_EXTENSIONS_DIR=$G/.torch_extensions
source "$G/.venv/bin/activate"
cd "$G"
# shellcheck disable=SC2086
python train.py --config-name "${CONFIG:-apps/colmap_3dgut.yaml}" \
    path="$DATA" out_dir="$OUT/runs" experiment_name="${S}_$POSES" \
    export_usd.enabled=true export_usd.format=nurec export_usd.apply_normalizing_transform=false \
    export_usd.path="$OUT/gaussians.usdz" export_ply.enabled=true export_ply.path="$OUT/gaussians.ply" \
    strategy.densify.clone_grad_threshold=0.00005 strategy.densify.split_grad_threshold=0.00005 \
    loss.use_aniso=true loss.lambda_aniso=0.01 loss.aniso_max_ratio=10 \
    loss.use_max_scale=true loss.lambda_max_scale=1.0 loss.max_scale=0.5 \
    ${EXTRA:-} 2>&1 | tee "$OUT/train.log"
echo "[gs] $OUT/gaussians.usdz"
