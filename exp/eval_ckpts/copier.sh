#!/usr/bin/env bash
cd /mnt/zhaozc_workspace/project/LQH/Drivor-orial/DrivoR-main
while true; do
  for run in steer_stage2 dplus; do
    base=exp/ke/${run}_vitb_8gpu_b16_12ep/11.12_10.30/lightning_logs
    mkdir -p "$base/keep_ckpts/checkpoints"
    for f in $(find "$base/version_0/checkpoints" -maxdepth 1 -name "epoch=*.ckpt" -mmin +1 2>/dev/null); do
      dst="$base/keep_ckpts/checkpoints/$(basename "$f")"
      if [ ! -f "$dst" ]; then cp "$f" "$dst.tmp" && mv "$dst.tmp" "$dst" && echo "$(date) copied $dst"; fi
    done
  done
  sleep 120
done
