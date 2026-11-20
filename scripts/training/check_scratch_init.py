"""Checks of the from-scratch pair (L with language, B0 without) before training.

Builds both models exactly as DrivoRSteerAgent does for MODEL=lang / MODEL=base in
run_scratch_8gpu_dlc.sh (seed 2, from_scratch=true, GridMask off, plan A), then checks:
  1. shared initialisation: every tensor of B0 equals the same tensor of L (same seed and
     construction order), L only adds the steer keys;
  2. nothing of D: LoRA B matrices are zero (ViT = plain DINOv2 at step 0);
  3. L's CA / connector / seg head equal the official SteerViT tensors;
  4. GridMask off in both;
  5. empty text: L's registers, proposals, pdm_score and trajectory equal B0's (bit-exact);
     with text the text rows change; CA receives [CLS | patches] = 3937 tokens;
  6. optimiser groups: DrivoR params at 0.0002*sqrt(128/64), steer params at --steer-lr.
With --model lang_frozen (MODEL=lang_frozen) the steer modules are frozen right after loading,
as DrivoRSteerAgent does, and check 6 becomes: every steer tensor has requires_grad=False and
the optimiser has exactly B0's single group (same parameter count, same lr).
"""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "nuplan-devkit")]
AGENT_CONFIG = ROOT / "navsim/planning/script/config/common/agent/drivoR_steer.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steervit-checkpoint", type=Path,
                        default=Path("/mnt/zhaozc_workspace/project/LQH/SteerViT_checkpoint/steervit_dinov2_base.pth"))
    parser.add_argument("--roberta", type=Path, default=Path("/mnt/zhaozc_workspace/project/LQH/RoBERTa"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--steer-lr", type=float, default=2.3e-5)
    parser.add_argument("--model", choices=("lang", "lang_frozen"), default="lang",
                        help="language model of run_scratch_8gpu_dlc.sh to check against B0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import pytorch_lightning as pl
    import torch
    from omegaconf import OmegaConf
    from navsim.agents.drivoR_steer.checkpoint_utils import initialise_training_weights, is_model_steer_key
    from navsim.agents.drivoR_steer.schedule_utils import build_param_groups
    from navsim.agents.drivoR_steer.steer_model import DrivoRSteerModel

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def build(lang: bool):
        cfg = OmegaConf.load(AGENT_CONFIG).config
        cfg.steer.enabled = lang
        cfg.steer.grid_mask = False
        cfg.steer.ca_on_registers = False
        cfg.steer.text_encoder = str(args.roberta)
        pl.seed_everything(args.seed, workers=True)          # as run_training_steer.py before instantiation
        model = DrivoRSteerModel(cfg)                        # DrivoRModel.__init__, then init_steer
        init = {"from_scratch": True, "drivor_checkpoint": "", "stage1_checkpoint": "",
                "steervit_checkpoint": str(args.steervit_checkpoint) if lang else ""}
        report = initialise_training_weights(model, OmegaConf.create(init), lang)
        if lang and frozen:
            report["frozen_values"] = model.image_backbone.freeze_steer_modules()   # as DrivoRSteerAgent
        return model, cfg, report

    frozen = args.model == "lang_frozen"

    lang, cfg, lang_report = build(True)
    base, _, base_report = build(False)
    report = {"init_lang": lang_report, "init_base": base_report}
    checks, failures = {}, []

    ls, bs = lang.state_dict(), base.state_dict()
    extra = sorted(set(ls) - set(bs))
    diff = [k for k in bs if not torch.equal(ls[k], bs[k])]
    checks["shared_init"] = {"base_tensors": len(bs), "different": diff[:10], "lang_extra": len(extra),
                             "lang_extra_all_steer": all(is_model_steer_key(k) for k in extra)}
    if diff or set(bs) - set(ls) or not checks["shared_init"]["lang_extra_all_steer"]:
        failures.append("shared_init")

    lora_b = {k: float(v.abs().max()) for k, v in bs.items() if ".linear_b_q." in k or ".linear_b_v." in k}
    checks["lora_b_zero"] = {"tensors": len(lora_b), "max_abs": max(lora_b.values()) if lora_b else None}
    if not lora_b or max(lora_b.values()) != 0.0:
        failures.append("lora_b_zero")

    official = torch.load(str(args.steervit_checkpoint), map_location="cpu", weights_only=False)["state_dict"]
    enc = lang.image_backbone
    mism = []
    for i in enc.steer_layers:
        for k, v in enc.vit.blocks[i].gated_cross_attn.state_dict().items():
            if not torch.equal(v, official[f"vision_model.trunk.blocks.{i}.gated_cross_attn.{k}"]):
                mism.append(f"block{i}.{k}")
    for name, mod in (("connector", enc.connector), ("lin_seg_head", enc.lin_seg_head)):
        mism += [f"{name}.{k}" for k, v in mod.state_dict().items() if not torch.equal(v, official[f"{name}.{k}"])]
    checks["steervit_loaded"] = {"mismatched": mism}
    if mism:
        failures.append("steervit_loaded")

    checks["grid_mask"] = {"lang": bool(lang.image_backbone.use_grid_mask), "base": bool(base.image_backbone.use_grid_mask)}
    if any(checks["grid_mask"].values()):
        failures.append("grid_mask")

    device = torch.device(args.device)
    lang.to(device).eval()
    base.to(device).eval()
    g = torch.Generator().manual_seed(0)
    B, N = 2, lang.num_cams
    W, H = cfg.image_size
    feats = {"image": torch.randn(B, N, 3, H, W, generator=g).to(device),
             "ego_status": torch.randn(B, 4, 11, generator=g).to(device)}

    def with_texts(rows):
        f = dict(feats)
        for n, key in enumerate(lang.text_keys):
            f[key] = [rows[b][n] for b in range(B)]
        return f

    seen = []
    hooks = [blk.gated_cross_attn.register_forward_pre_hook(lambda m, inp: seen.append(int(inp[0].shape[1])))
             for blk in enc.vit.blocks if blk.gated_cross_attn is not None]
    keys = ("proposals", "pdm_score", "trajectory")
    T = cfg.num_scene_tokens
    with torch.no_grad():
        ref = base(feats)
        ref_tok = base.image_backbone(feats["image"], base.scene_embeds.repeat(B, 1, 1, 1))
        out = lang(with_texts([[""] * N for _ in range(B)]))
        enc._steer_ctx = lang.encode_text([""] * (B * N), torch.zeros(B * N, dtype=torch.bool))
        tok = enc(feats["image"], lang.scene_embeds.repeat(B, 1, 1, 1))
        enc._steer_ctx = None
        md = lambda a, b: float((a.float() - b.float()).abs().max())  # noqa: E731
        checks["empty_text_equals_base"] = {"registers": md(tok, ref_tok), **{k: md(out[k], ref[k]) for k in keys}}
        if max(checks["empty_text_equals_base"].values()) != 0.0:
            failures.append("empty_text_equals_base")

        rows = [[""] * N for _ in range(B)]
        rows[0][0] = "a white sedan ahead in the ego lane"
        seen.clear()
        enc._steer_ctx = lang.encode_text([t for r in rows for t in r],
                                          torch.tensor([bool(t) for r in rows for t in r]))
        tok_t = enc(feats["image"], lang.scene_embeds.repeat(B, 1, 1, 1)).reshape(B, N, T, -1)
        enc._steer_ctx = None
        ref_rows = ref_tok.reshape(B, N, T, -1)
        others = max(md(tok_t[b, n], ref_rows[b, n]) for b in range(B) for n in range(N) if (b, n) != (0, 0))
        checks["text_routing"] = {"text_row_changed": md(tok_t[0, 0], ref_rows[0, 0]), "other_rows_max_diff": others,
                                  "ca_query_tokens": seen}
        if others != 0.0 or checks["text_routing"]["text_row_changed"] == 0.0 or \
                seen != [48 * 82 + 1] * len(enc.steer_layers):
            failures.append("text_routing")
    for h in hooks:
        h.remove()

    lr = 0.0002 * math.sqrt(16 * 8 / 64)
    steer_ids = {id(p) for p in enc.steer_parameters()}
    groups = build_param_groups(lang.named_parameters(), steer_ids, lr, 1.0, 0.01, None if frozen else args.steer_lr)
    checks["optimizer_groups"] = {g["name"]: {"lr": g["lr"], "params": sum(p.numel() for p in g["params"])}
                                  for g in groups}
    base_groups = build_param_groups(base.named_parameters(), set(), lr, 1.0, 0.01, None)
    checks["optimizer_groups_base"] = {g["name"]: {"lr": g["lr"], "params": sum(p.numel() for p in g["params"])}
                                       for g in base_groups}
    if checks["optimizer_groups"]["drivor"]["params"] != checks["optimizer_groups_base"]["drivor"]["params"]:
        failures.append("optimizer_groups")
    elif frozen:
        trainable_steer = [n for n, p in enc.named_parameters() if enc.is_steer_key(n) and p.requires_grad]
        checks["steer_frozen"] = {"trainable_steer_tensors": trainable_steer[:10],
                                  "steer_tensors": sum(1 for n, _ in enc.named_parameters() if enc.is_steer_key(n))}
        if trainable_steer or set(checks["optimizer_groups"]) != {"drivor"} or \
                checks["optimizer_groups"] != checks["optimizer_groups_base"]:
            failures.append("steer_frozen")
    elif abs(checks["optimizer_groups"]["steer"]["lr"] - args.steer_lr) > 1e-12:
        failures.append("optimizer_groups")

    report.update(checks=checks, failures=failures, passed=not failures)
    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")
    if failures:
        raise SystemExit(f"FAILED: {failures}")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
