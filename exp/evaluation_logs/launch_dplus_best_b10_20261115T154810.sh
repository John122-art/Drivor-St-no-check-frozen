#!/usr/bin/env bash
set -eo pipefail
REPO_ROOT=/mnt/zhaozc_workspace/project/LQH/Drivor-orial/DrivoR-main
RUN_NAME=dplus_best_navtest_1gpu_b10_20261115T154810
cd "$REPO_ROOT"
LOG_PATH="$REPO_ROOT/exp/evaluation_logs/$RUN_NAME.log"
PID_PATH="$REPO_ROOT/exp/evaluation_logs/$RUN_NAME.pid"
STATUS_PATH="$REPO_ROOT/exp/evaluation_logs/$RUN_NAME.status"
OUTPUT_PATH="$REPO_ROOT/exp/$RUN_NAME"
if [[ "${1:-}" != --worker ]]; then
    [[ $# == 0 && ! -e "$LOG_PATH" && ! -e "$PID_PATH" && ! -e "$OUTPUT_PATH" ]]
    nohup setsid bash "$REPO_ROOT/exp/evaluation_logs/launch_dplus_best_b10_20261115T154810.sh" --worker > "$LOG_PATH" 2>&1 < /dev/null &
    EVAL_PID=$!
    printf '%s\n' "$EVAL_PID" > "$PID_PATH"
    sleep 3
    kill -0 "$EVAL_PID"
    printf 'PID=%s\nLOG=%s\nOUTPUT=%s\n' "$EVAL_PID" "$LOG_PATH" "$OUTPUT_PATH"
    exit 0
fi
trap 'result=$?; if (( result != 0 )); then printf "failed exit=%s\n" "$result" > "$STATUS_PATH"; fi' EXIT
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
command=(/mnt/zhaozc_workspace/envs/drivorpython/bin/python -B scripts/evaluation/evaluate_drivor_steer.py
    --checkpoint "$REPO_ROOT/exp/ke/dplus_vitb_8gpu_b16_12ep/11.12_10.30/lightning_logs/version_0/checkpoints/best-epoch=11-step=9672.ckpt"
    --dataset "$OPENSCENE_DATA_ROOT"
    --metric-cache /mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/drivor_official_navtest_full_cache_20261007/metric_cache
    --limit 0 --batch-size 10)
printf 'preflight_cpu\n' > "$STATUS_PATH"
printf 'D+ full-navtest FP32 batch_size=10. Validating inputs and checkpoint on CPU: '
date -u '+%Y-%m-%d %H:%M:%S UTC'
CUDA_VISIBLE_DEVICES= "${command[@]}" --output "${OUTPUT_PATH}_cpu_check" --check-model
printf 'waiting_for_pid=90882\n' > "$STATUS_PATH"
printf 'CPU validation passed. Waiting for Steer-DrivoR PID 90882 to release the single GPU.\n'
while [[ -r /proc/90882/cmdline ]] && rg -a -F -q 'steer_stage2_capatch_best_navtest_1gpu_b26_20261115T143848_r2' /proc/90882/cmdline; do
    sleep 30
done
printf 'running\n' > "$STATUS_PATH"
printf 'Previous evaluation exited. Starting D+ batch_size=10: '
date -u '+%Y-%m-%d %H:%M:%S UTC'
"${command[@]}" --output "$OUTPUT_PATH"
printf 'complete\n' > "$STATUS_PATH"
date -u '+D+ evaluation complete: %Y-%m-%d %H:%M:%S UTC'
