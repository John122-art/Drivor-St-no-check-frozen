"""DrivoRAgent with language steering (stage 2) and the D+ control (steer.enabled=false).

Differences from DrivoRAgent:
  * the model is converted in place to DrivoRSteerModel (no second ViT construction);
  * for training (checkpoint_path == ""), weights are initialised from checkpoint D and,
    if steering is enabled, from the stage-1 checkpoint or the official SteerViT file;
    with init_args.from_scratch=true nothing of D is loaded (fresh DrivoR initialisation)
    and the steer modules come from the official SteerViT file;
  * loss = original DrivoR loss + ground_loss_weight * s * L_ground;
  * AdamW parameter groups: DrivoR parameters at d_lr_scale, steer modules at
    steer_lr_scale (both relative to the original auto-scaled learning rate), or at the
    absolute lr_args.steer_lr when it is set;
  * scheduler_args.exact_steps=true: the schedule length is the real number of optimiser
    steps (set by run_training_steer.py from the dataset size) instead of dataset_size;
  * keep_all_epochs=true: every epoch's checkpoint (epoch=X-step=Y.ckpt) is kept;
  * a callback ramps s linearly from 0 to 1 over gate_ramp_fraction of training;
  * steer.grounding=false: no grounding loss (the dataset then reads no grounding files);
  * steer.freeze_steer_modules=true: CA (with its tanh gates), connector and seg head stay at
    their initial (official SteerViT) weights; they are left out of the optimiser and a
    callback checks after every training epoch that they did not change.
The DrivoR scheduler (LinearLR ramp + cosine) is reused unchanged.
"""
import math
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint

from navsim.agents.drivoR.drivor_agent import DrivoRAgent
from .checkpoint_utils import initialise_training_weights
from .config_utils import cfg_value
from .schedule_utils import build_param_groups, total_scheduler_steps
from .steer_loss import ddp_view_mean, ground_loss_terms
from .steer_model import DrivoRSteerModel


def _rank_zero() -> bool:
    import torch.distributed as dist
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


class GateRampCallback(pl.Callback):
    """s = min(1, global_step / ramp_steps); ramp_steps <= 0 means s = 1 throughout."""

    def __init__(self, ramp_steps: int):
        super().__init__()
        self.ramp_steps = int(ramp_steps)

    def value(self, step: int) -> float:
        if self.ramp_steps <= 0:
            return 1.0
        return min(1.0, step / self.ramp_steps)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        model = pl_module.agent._drivor_model
        model.steer_gate_scale.fill_(self.value(trainer.global_step))


class StepsCheckCallback(pl.Callback):
    """With exact_steps: confirm that the scheduler's steps per epoch equal Lightning's."""

    def __init__(self, steps_per_epoch: int):
        super().__init__()
        self.steps_per_epoch = int(steps_per_epoch)

    def on_train_start(self, trainer, pl_module):
        actual = int(trainer.num_training_batches) // int(trainer.accumulate_grad_batches)
        status = "OK" if actual == self.steps_per_epoch else "WARNING: MISMATCH"
        if _rank_zero():
            print(f"[DrivoRSteerAgent] scheduler steps/epoch {self.steps_per_epoch}, "
                  f"Lightning steps/epoch {actual}: {status}", flush=True)


class FrozenSteerCheckCallback(pl.Callback):
    """steer.freeze_steer_modules: after every training epoch the steer tensors must still equal
    those at the start of training (checked on rank 0, reported in the log)."""

    def __init__(self):
        super().__init__()
        self.reference = None

    def on_train_start(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        enc = pl_module.agent._drivor_model.image_backbone
        self.reference = {n: p.detach().cpu().clone() for n, p in enc.named_parameters() if enc.is_steer_key(n)}

    def on_train_epoch_end(self, trainer, pl_module):
        if self.reference is None:
            return
        current = dict(pl_module.agent._drivor_model.image_backbone.named_parameters())
        changed = [n for n, v in self.reference.items()
                   if current[n].requires_grad or not torch.equal(current[n].detach().cpu(), v)]
        status = f"ERROR: {len(changed)} changed or trainable, e.g. {changed[:3]}" if changed else "unchanged"
        print(f"[DrivoRSteerAgent] epoch {trainer.current_epoch}: frozen steer tensors "
              f"({len(self.reference)}) {status}", flush=True)


class DrivoRSteerAgent(DrivoRAgent):
    def __init__(
            self,
            config,
            lr_args: dict,
            checkpoint_path: str = None,
            loss: nn.Module = None,
            progress_bar: bool = True,
            scheduler_args: dict = None,
            batch_size: int = 64,
            num_gpus: int = 1,
            init_args: dict = None,
            keep_all_epochs: bool = False,
    ):
        super().__init__(config, lr_args, checkpoint_path, loss, progress_bar, scheduler_args, batch_size, num_gpus)
        self._drivor_model.__class__ = DrivoRSteerModel
        self._drivor_model.init_steer(config)
        self.steer_enabled = self._drivor_model.steer_enabled
        self.ground_loss_weight = float(cfg_value(config.steer, "ground_loss_weight", 0.5))
        self.gate_ramp_fraction = float(cfg_value(config.steer, "gate_ramp_fraction", 0.0))
        # Missing keys = earlier behaviour (grounding loss on, steer modules trainable).
        self.grounding = self.steer_enabled and bool(cfg_value(config.steer, "grounding", True))
        self.freeze_steer = self.steer_enabled and bool(cfg_value(config.steer, "freeze_steer_modules", False))
        self._init_args = init_args or {}
        self.keep_all_epochs = bool(keep_all_epochs)
        self.exact_steps = bool(cfg_value(scheduler_args, "exact_steps", False)) if scheduler_args is not None else False
        self._steps_per_epoch = None
        self.init_report = {}
        if self._checkpoint_path == "":
            self.init_report = initialise_training_weights(self._drivor_model, self._init_args, self.steer_enabled)
            if _rank_zero():
                print(f"[DrivoRSteerAgent] initialisation: {self.init_report}", flush=True)
        if self.freeze_steer:
            # Also when resuming / evaluating: requires_grad is not stored in checkpoints, and the
            # optimiser groups must match those of the run that wrote the checkpoint.
            frozen = self._drivor_model.image_backbone.freeze_steer_modules()
            if _rank_zero():
                print(f"[DrivoRSteerAgent] steer modules frozen ({frozen} values); grounding loss "
                      f"{'on' if self.grounding else 'off'}", flush=True)
        elif self.steer_enabled and not self.grounding:
            # Trainable CA / connector without grounding loss: the seg head would get no gradient
            # (and DDP would report it as unused), so it alone is frozen.
            self._drivor_model.image_backbone.lin_seg_head.requires_grad_(False)

    def set_steps_per_epoch(self, steps: int):
        """Called by run_training_steer.py when scheduler_args.exact_steps is true."""
        if int(steps) <= 0:
            raise ValueError(f"steps per epoch must be positive, got {steps}")
        self._steps_per_epoch = int(steps)

    # ------------------------------------------------------------------ loss
    def compute_loss(self, features, targets, pred) -> Dict:
        loss_dict = super().compute_loss(features, targets, pred)
        if not self.steer_enabled:
            return loss_dict
        if not self.grounding:
            loss_dict["steer_gate_scale"] = self._drivor_model.steer_gate_scale.detach().clone()
            loss_dict["steer_text_dropped"] = pred["steer_text_dropped"].float().mean()
            return loss_dict
        logits = pred["steer_ground_logits"]
        B, N, P = logits.shape
        valid = targets["steer_ground_valid"].to(torch.bool) & pred["steer_text_active"]
        loss_sum, count, pmass = ground_loss_terms(
            logits.reshape(B * N, P), targets["steer_ground_target"].reshape(B * N, P),
            valid.reshape(-1), pred["steer_patch_keep"])
        ground = ddp_view_mean(loss_sum, count)
        scale = self._drivor_model.steer_gate_scale.to(ground.dtype)
        loss_dict["loss"] = loss_dict["loss"] + self.ground_loss_weight * scale * ground
        loss_dict["steer_ground_loss"] = ground.detach()
        loss_dict["steer_ground_views"] = count.detach().float()
        loss_dict["steer_pmass"] = pmass.detach().mean() if pmass.numel() else torch.zeros((), device=logits.device)
        loss_dict["steer_gate_scale"] = scale.detach()
        loss_dict["steer_text_dropped"] = pred["steer_text_dropped"].float().mean()
        return loss_dict

    # ------------------------------------------------------------------ optimisation
    def _total_steps(self) -> int:
        if self.exact_steps and self._steps_per_epoch is None:
            raise RuntimeError("scheduler_args.exact_steps=true but steps per epoch were not set "
                               "(use navsim/planning/script/run_training_steer.py)")
        return total_scheduler_steps(self.scheduler_args.dataset_size, self.batch_size * self.num_gpus,
                                     self.scheduler_args.num_epochs, self._steps_per_epoch)

    def get_optimizers(self):
        global_batchsize = self.batch_size * self.num_gpus
        if self._lr_args["name"] != "AdamW":
            raise NotImplementedError("DrivoRSteerAgent uses AdamW as the DrivoR ViT-B training did")
        lr = self._lr_args["base_lr"] * math.sqrt(global_batchsize / self._lr_args["base_batch_size"])
        d_scale = float(cfg_value(self._lr_args, "d_lr_scale", 1.0))
        steer_scale = float(cfg_value(self._lr_args, "steer_lr_scale", d_scale))
        steer_lr = cfg_value(self._lr_args, "steer_lr", None)

        encoder = self._drivor_model.image_backbone
        steer_ids = {id(p) for p in encoder.steer_parameters()} if self.steer_enabled else set()
        groups = build_param_groups(self._drivor_model.named_parameters(), steer_ids, lr, d_scale, steer_scale,
                                    None if steer_lr is None else float(steer_lr))
        optimizer = torch.optim.AdamW(groups, lr=lr * d_scale)
        if _rank_zero():
            desc = ", ".join(f"{g['name']}: peak lr {g['lr']:.4e} ({sum(p.numel() for p in g['params'])} params)"
                             for g in groups)
            print(f"[DrivoRSteerAgent] AdamW: auto-scaled base lr {lr:.4e}; {desc}", flush=True)

        if self.scheduler_args is None:
            return [optimizer]
        # Ramp + cosine, as DrivoRAgent.get_optimizers.
        T_max = self._total_steps()
        if _rank_zero():
            print(f"[DrivoRSteerAgent] scheduler: {T_max} steps "
                  f"({'exact' if self._steps_per_epoch is not None else 'from dataset_size'}), "
                  f"ramp {int(T_max * 0.1)}", flush=True)
        T_max_ramp = int(T_max * 0.1)
        scheduler_ramp = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, total_iters=T_max_ramp)
        T_max_cosine = T_max - T_max_ramp
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max_cosine,
            eta_min=0.0, last_epoch=-1
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[scheduler_ramp, scheduler_cosine],
            milestones=[T_max_ramp],
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def get_training_callbacks(self):
        callbacks = super().get_training_callbacks()
        if self.keep_all_epochs:
            # DrivoRAgent's unmonitored ModelCheckpoint(save_last=True) writes epoch=X-step=Y.ckpt
            # every epoch and deletes the previous one (save_top_k=1); keep them all instead.
            # (A second unmonitored ModelCheckpoint would share its Lightning state key.)
            plain = [cb for cb in callbacks if isinstance(cb, ModelCheckpoint) and cb.monitor is None]
            if len(plain) != 1:
                raise RuntimeError(f"Expected one unmonitored ModelCheckpoint, found {len(plain)}")
            plain[0].save_top_k = -1
        if self.exact_steps:
            callbacks.append(StepsCheckCallback(self._steps_per_epoch))
        if self.steer_enabled:
            ramp_steps = int(self.gate_ramp_fraction * self._total_steps()) if self.scheduler_args is not None else 0
            callbacks.append(GateRampCallback(ramp_steps))
        if self.freeze_steer:
            callbacks.append(FrozenSteerCheckCallback())
        return callbacks
