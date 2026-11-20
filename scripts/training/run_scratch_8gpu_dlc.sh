#!/usr/bin/env bash
# From-scratch comparison on DLC (8 devices): language model L vs language-free baseline B0.
# Both start like the checkpoint-D training (DINOv2 ViT-B weights, new LoRA, random registers / neck /
# decoders; nothing of D is loaded) and use D's recipe: 25 epochs, batch 16 x 8, base_lr 2e-4, seed 2,
# same DrivoR overrides as run_drivor_vitb_8gpu_dlc.sh. Differences from D: GridMask off and the
# scheduler covers the real number of optimiser steps (exact_steps). Every epoch's checkpoint is kept.
#   MODEL=lang bash $0         # L : + SteerViT CA/connector/seg head (official weights), plan A, grounding loss
#   MODEL=lang_frozen bash $0  # L-frozen: same, but CA/connector/seg head frozen at the official SteerViT
#                              #   weights (tanh gates included), no grounding loss, gate scale s = 1 from step 0
#   MODEL=base bash $0         # B0: no language
#   ... bash $0 --dry-run
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
REFERENCE_ROOT="${REFERENCE_ROOT:-/mnt/zhaozc_workspace/project/LQH/DrivoR-main-copy}"
ENV_ROOT="${ENV_ROOT:-/mnt/zhaozc_workspace/envs/drivorpython}"
MODEL="${MODEL:-}"
BATCH_SIZE="${BATCH_SIZE:-16}"
RESUME_TRAINING="${RESUME_TRAINING:-false}"
case "$RESUME_TRAINING" in true|false) ;; *) echo "RESUME_TRAINING must be true or false" >&2; exit 2 ;; esac
MAX_EPOCHS="${MAX_EPOCHS:-25}"
STEER_LR="${STEER_LR:-2.3e-5}"            # absolute peak lr of CA / connector / seg head (MODEL=lang only)
STEERVIT_CHECKPOINT="${STEERVIT_CHECKPOINT:-/mnt/zhaozc_workspace/project/LQH/SteerViT_checkpoint/steervit_dinov2_base.pth}"
ROBERTA_PATH="${ROBERTA_PATH:-/mnt/zhaozc_workspace/project/LQH/RoBERTa}"
STAGE2_BUNDLE="${STAGE2_BUNDLE:-/mnt/zhaozc_workspace/project/LQH/Driv-calude_data/DrivoR-main/outputs/stage2_focus_v2_full_p3}"
case "$MODEL" in
    lang) EXPERIMENT_NAME="${EXPERIMENT_NAME:-scratch_lang_capatch_nogm_vitb_8gpu_b${BATCH_SIZE}_${MAX_EPOCHS}ep}" ;;
    lang_frozen) EXPERIMENT_NAME="${EXPERIMENT_NAME:-scratch_langfrozen_noground_capatch_nogm_vitb_8gpu_b${BATCH_SIZE}_${MAX_EPOCHS}ep}" ;;
    base) EXPERIMENT_NAME="${EXPERIMENT_NAME:-scratch_base_nogm_vitb_8gpu_b${BATCH_SIZE}_${MAX_EPOCHS}ep}" ;;
    *) echo "Set MODEL=lang, MODEL=lang_frozen or MODEL=base" >&2; exit 2 ;;
esac

export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$REPO_ROOT/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$REFERENCE_ROOT/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$OPENSCENE_DATA_ROOT/maps}"
DRIVOR_TRAIN_METRIC_CACHE_PATH="${DRIVOR_TRAIN_METRIC_CACHE_PATH:-/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/official_metric_cache_navtrain_20261005/metric_cache}"
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "BATCH_SIZE must be a positive integer" >&2; exit 2; }
for v in "$STEERVIT_CHECKPOINT" "$ROBERTA_PATH" "$STAGE2_BUNDLE"; do
    [[ "$v" != *"'"* ]] || { echo "Paths must not contain single quotes: $v" >&2; exit 2; }
done
DRY_RUN=false
if [[ "${1:-}" == --dry-run ]]; then DRY_RUN=true; shift; fi

for input_dir in "$OPENSCENE_DATA_ROOT/navsim_logs/trainval" "$OPENSCENE_DATA_ROOT/sensor_blobs/trainval" "$NUPLAN_MAPS_ROOT"; do
    [[ -d "$input_dir" ]] || { echo "Missing NAVSIM input directory: $input_dir" >&2; exit 1; }
done

# Common to L and B0: from scratch, no GridMask, exact schedule, all epoch checkpoints.
model_args=("agent.init_args.from_scratch=true"
    "agent.init_args.drivor_checkpoint=''"
    "agent.init_args.stage1_checkpoint=''"
    "agent.config.steer.grid_mask=false"
    "agent.scheduler_args.exact_steps=true"
    "agent.keep_all_epochs=true"
    "agent.lr_args.d_lr_scale=1.0")
if [[ "$MODEL" == lang || "$MODEL" == lang_frozen ]]; then
    for f in "$STAGE2_BUNDLE/train/dataset_index.jsonl" "$STAGE2_BUNDLE/val/dataset_index.jsonl" \
             "$STAGE2_BUNDLE/train/COMPLETE.json" "$STAGE2_BUNDLE/val/COMPLETE.json" "$STEERVIT_CHECKPOINT"; do
        [[ -f "$f" ]] || { echo "Missing file: $f" >&2; exit 1; }
    done
    [[ -d "$ROBERTA_PATH" ]] || { echo "Missing RoBERTa directory: $ROBERTA_PATH" >&2; exit 1; }
    model_args+=("agent.config.steer.enabled=true"
        "agent.config.steer.ca_on_registers=false"
        "agent.config.steer.gate_ramp_fraction=0.0"
        "agent.config.steer.text_encoder='$ROBERTA_PATH'"
        "agent.init_args.steervit_checkpoint='$STEERVIT_CHECKPOINT'"
        "+steer_data.train_index='$STAGE2_BUNDLE/train/dataset_index.jsonl'"
        "+steer_data.val_index='$STAGE2_BUNDLE/val/dataset_index.jsonl'")
    if [[ "$MODEL" == lang ]]; then
        model_args+=("agent.lr_args.steer_lr=$STEER_LR")
    else
        # Language modules stay at the official weights; only LoRA / registers / neck / decoders
        # train, with exactly B0's optimiser group. Texts are read, grounding files are not.
        model_args+=("agent.config.steer.grounding=false"
            "agent.config.steer.freeze_steer_modules=true")
    fi
else
    model_args+=("agent.config.steer.enabled=false"
        "agent.init_args.steervit_checkpoint=''")
fi

metric_cache_link="$NAVSIM_EXP_ROOT/train_metric_cache"
if [[ ! -e "$metric_cache_link" && ! -L "$metric_cache_link" ]]; then
    [[ -d "$DRIVOR_TRAIN_METRIC_CACHE_PATH/metadata" ]] || { echo "Missing training metric-cache metadata" >&2; exit 1; }
    mkdir -p "$NAVSIM_EXP_ROOT"
    ln -s "$DRIVOR_TRAIN_METRIC_CACHE_PATH" "$metric_cache_link"
fi
[[ -d "$metric_cache_link/metadata" ]] || { echo "Invalid training metric cache: $metric_cache_link" >&2; exit 1; }

set +u
if [[ -f /usr/local/PPU_SDK/envsetup.sh ]]; then source /usr/local/PPU_SDK/envsetup.sh; fi
source "$ENV_ROOT/bin/activate"
set -u
export NAVSIM_DEVKIT_ROOT="$REPO_ROOT"
export NAVSIM_EXP_ROOT OPENSCENE_DATA_ROOT NUPLAN_MAPS_ROOT
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/nuplan-devkit${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/weights/hf_cache}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/drivor_steer_matplotlib}"
cd "$REPO_ROOT"

case "$MODEL" in lang) steer_desc="$STEER_LR" ;; lang_frozen) steer_desc="frozen (official SteerViT), no grounding loss" ;; *) steer_desc="-" ;; esac
printf 'MODEL=%s, experiment=%s, devices=8, batch/device=%s, epochs=%s, steer modules: %s\n' \
    "$MODEL" "$EXPERIMENT_NAME" "$BATCH_SIZE" "$MAX_EPOCHS" "$steer_desc"
printf 'Resume training=%s (false: start at epoch 0; true: restore checkpoint).\n' "$RESUME_TRAINING"

# DrivoR overrides below are those of run_drivor_vitb_8gpu_dlc.sh (checkpoint D).
command=(python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_steer.py" \
    agent=drivoR_steer \
    "+resume_training=$RESUME_TRAINING" \
    "experiment_name=$EXPERIMENT_NAME" \
    train_test_split=navtrain \
    cache_path=null \
    use_cache_without_dataset=false \
    "trainer.params.max_epochs=$MAX_EPOCHS" \
    +trainer.params.devices=8 \
    trainer.params.num_nodes=1 \
    trainer.params.strategy=ddp \
    trainer.params.accumulate_grad_batches=1 \
    dataloader.params.num_workers=2 \
    dataloader.params.prefetch_factor=1 \
    "dataloader.params.batch_size=$BATCH_SIZE" \
    "agent.batch_size=$BATCH_SIZE" \
    agent.lr_args.name=AdamW \
    agent.lr_args.base_lr=0.0002 \
    agent.lr_args.base_batch_size=64 \
    agent.num_gpus=8 \
    agent.progress_bar=false \
    agent.config.refiner_ls_values=0.0 \
    agent.config.image_backbone.focus_front_cam=false \
    agent.config.one_token_per_traj=true \
    agent.config.refiner_num_heads=1 \
    agent.config.tf_d_model=256 \
    agent.config.tf_d_ffn=1024 \
    agent.config.area_pred=false \
    agent.config.agent_pred=false \
    agent.config.ref_num=4 \
    agent.loss.prev_weight=0.0 \
    agent.config.long_trajectory_additional_poses=2 \
    "${model_args[@]}" \
    seed=2 \
    "$@")
if [[ "$DRY_RUN" == true ]]; then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
exec "${command[@]}"
