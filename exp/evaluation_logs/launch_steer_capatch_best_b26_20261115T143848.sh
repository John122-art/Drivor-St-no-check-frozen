#!/usr/bin/env bash
set -eo pipefail
REPO_ROOT=/mnt/zhaozc_workspace/project/LQH/Drivor-orial/DrivoR-main
RUN_NAME=steer_stage2_capatch_best_navtest_1gpu_b26_20261115T143848_r2
cd "$REPO_ROOT"
if [[ -f /usr/local/PPU_SDK/envsetup.sh ]]; then source /usr/local/PPU_SDK/envsetup.sh; fi
source /mnt/zhaozc_workspace/envs/drivorpython/bin/activate
export CUDA_VISIBLE_DEVICES=0
export NAVSIM_DEVKIT_ROOT="$REPO_ROOT" NAVSIM_EXP_ROOT="$REPO_ROOT/exp"
export OPENSCENE_DATA_ROOT=/mnt/zhaozc_workspace/project/LQH/DrivoR-main-copy/dataset
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps" NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/nuplan-devkit${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$REPO_ROOT/weights/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR=/tmp/drivor_vitb_matplotlib
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
LOG_PATH="$REPO_ROOT/exp/evaluation_logs/$RUN_NAME.log"
PID_PATH="$REPO_ROOT/exp/evaluation_logs/$RUN_NAME.pid"
OUTPUT_PATH="$REPO_ROOT/exp/$RUN_NAME"
[[ ! -e "$LOG_PATH" && ! -e "$PID_PATH" && ! -e "$OUTPUT_PATH" ]]
nohup setsid /mnt/zhaozc_workspace/envs/drivorpython/bin/python -B scripts/evaluation/evaluate_drivor_steer.py \
  --checkpoint "$REPO_ROOT/exp/ke/steer_stage2_capatch_vitb_8gpu_b16_12ep/11.13_05.07/lightning_logs/version_0/checkpoints/best-epoch=10-step=8866.ckpt" \
  --dataset "$OPENSCENE_DATA_ROOT" \
  --metric-cache /mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/drivor_official_navtest_full_cache_20261007/metric_cache \
  --output "$OUTPUT_PATH" --limit 0 --batch-size 26 \
  --text-index /mnt/zhaozc_workspace/project/LQH/Driv-calude_data/DrivoR-main/outputs/stage2_focus_v2_full_p3/test/dataset_index.jsonl \
  --text-mode correct --shuffle-seed 0 > "$LOG_PATH" 2>&1 < /dev/null &
EVAL_PID=$!
printf '%s\n' "$EVAL_PID" > "$PID_PATH"
sleep 3
kill -0 "$EVAL_PID"
printf 'PID=%s\nLOG=%s\nOUTPUT=%s\n' "$EVAL_PID" "$LOG_PATH" "$OUTPUT_PATH"
