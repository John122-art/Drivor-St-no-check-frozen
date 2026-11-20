"""Scheduler length and optimiser parameter groups for DrivoRSteerAgent."""
import math
from typing import Dict, List, Optional


def lightning_steps_per_epoch(num_samples: int, world_size: int, batch_size: int) -> int:
    """Optimiser steps per epoch of the train DataLoader built in run_training_steer.py
    (shuffle=True, drop_last=True) under Lightning DDP with accumulate_grad_batches=1:
    Lightning's DistributedSampler (drop_last=False) gives ceil(N / world) samples per rank
    and the batch sampler drops the last incomplete batch."""
    if num_samples <= 0 or world_size <= 0 or batch_size <= 0:
        raise ValueError(f"Invalid sizes: samples={num_samples}, world={world_size}, batch={batch_size}")
    return math.ceil(num_samples / world_size) // batch_size


def total_scheduler_steps(dataset_size: int, global_batch_size: int, num_epochs: int,
                          steps_per_epoch: Optional[int] = None) -> int:
    """Steps covered by the ramp + cosine schedule. Without steps_per_epoch this is the
    DrivoRAgent formula (dataset_size from the YAML, 85000 for checkpoint D)."""
    if steps_per_epoch is not None:
        return int(steps_per_epoch) * int(num_epochs)
    return int(math.ceil(dataset_size / global_batch_size) * num_epochs)


def build_param_groups(named_parameters, steer_ids, lr: float, d_lr_scale: float, steer_lr_scale: float,
                       steer_lr: Optional[float] = None) -> List[Dict]:
    """AdamW groups: DrivoR parameters at lr * d_lr_scale; steer modules (CA, connector,
    seg head) at the absolute steer_lr if given, else lr * steer_lr_scale. Frozen parameters
    are left out."""
    d_params, steer_params = [], []
    for _, p in named_parameters:
        if not p.requires_grad:
            continue
        (steer_params if id(p) in steer_ids else d_params).append(p)
    groups = [{"params": d_params, "lr": lr * d_lr_scale, "name": "drivor"}]
    if steer_params:
        groups.append({"params": steer_params, "lr": float(steer_lr) if steer_lr is not None else lr * steer_lr_scale,
                       "name": "steer"})
    return groups
