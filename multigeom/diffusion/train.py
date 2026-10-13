"""
Stage 3: Train a conditional DDPM on TinyImageNet, conditioned on the
synthetic responses produced in Stage 2.

Training recipe (paper appendix "Diffusion model architecture and training"):
  - DDPM linear β schedule, T = 1000, ε-prediction objective.
  - AdamW with lr 1e-4, weight decay 1e-4, cosine schedule with 500-step warmup.
  - Batch size 128 in bfloat16, gradient clipping at norm 1.0.
  - 200 epochs (the checkpoint used in all figures).
  - Conditioning vector r̃ = r + σ ξ (σ = 0.22) injected per-step.
  - Classifier-free guidance dropout at rate 0.1: with that probability,
    the conditioning vector is zeroed out so the network learns both
    p(ε | x_t, r̃) and p(ε | x_t, ∅).

Usage:
    python -m multigeom.diffusion.train \\
        --config diffusion \\
        --responses runs/responses/inception/V1/train_responses.pt \\
        --out-dir runs/diffusion/inception/V1
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


from ..config import load_config
from ..data import CANONICAL_TRANSFORM, TinyImageNetDataset
from ..utils import count_parameters, get_device, provenance, set_seed
from .model import build_diffusion_model, build_scheduler


def parse_args(argv=None):
    """Command-line arguments of `python -m multigeom.diffusion.train`."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--responses", required=True,
                   help="Stage-2 train_responses.pt produced by `multigeom.diffusion.responses`")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-every", type=int, default=20,
                   help="checkpoint period in epochs")
    p.add_argument("--resume", action="store_true",
                   help="continue from <out-dir>/last.pt if it exists")
    return p.parse_args(argv)


class TinyImageNetWithResponses(torch.utils.data.Dataset):
    """Pair TinyImageNet images with the precomputed synthetic response vectors."""

    def __init__(self, image_dataset: TinyImageNetDataset, responses: torch.Tensor):
        if len(image_dataset) != responses.shape[0]:
            raise ValueError(
                f"Dataset size ({len(image_dataset)}) != responses size ({responses.shape[0]}); "
                "ensure the response file was produced by `multigeom.diffusion.responses` "
                "for the same TinyImageNet split, in sorted-class / sorted-file order."
            )
        self.image_dataset = image_dataset
        self.responses = responses

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        return self.image_dataset[idx], self.responses[idx]


def cosine_with_warmup(step: int, warmup: int, total: int) -> float:
    """Multiplicative LR factor: linear warmup, then cosine decay to 0."""
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_diffusion(
    cfg,
    image_dataset,
    responses: torch.Tensor,
    out_dir,
    *,
    device=None,
    num_workers: int = 4,
    save_every: int = 20,
    resume: bool = False,
    metadata: dict | None = None,
    prov: dict | None = None,
    log=print,
    progress: bool = True,
):
    """
    Train the conditional diffusion model of an encoder, and write `final_model.pt` (rewritten after every epoch),
    `train_log.json`, and `last.pt` (to resume an interrupted run with `resume=True`) to `out_dir`.

    Args:
        cfg: the diffusion configuration (`DiffusionConfig.for_encoder(encoder)` sets the conditioning width). The
            recipe (epochs, batch size, learning rate, ...) is `cfg.train`; copy and edit `cfg` for a shorter run.
        image_dataset: a dataset of images [1, 64, 64] in [-1, 1] (for example `TinyImageNetDataset`).
        responses: [len(image_dataset), n_neurons] responses of the encoder to those images
            (see `multigeom.diffusion.responses.precompute_responses`).
        out_dir: output folder.
        metadata: extra entries saved in the checkpoint (for example the encoder name and area).
        log: where progress messages go (`print`, a function taking a string, or None for silence).
        progress: show a progress bar over the batches of each epoch.

    Returns:
        The loss history, `{"epoch": [...], "loss": [...]}`.
    """
    device = get_device(device) if not isinstance(device, torch.device) else device
    set_seed(cfg.train.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = metadata or {}
    prov = prov if prov is not None else provenance({"diffusion": "diffusion"}, seed=cfg.train.seed)
    say = log or (lambda *_: None)

    n_neurons = int(responses.shape[1])
    if n_neurons != cfg.conditioning.mlp_layers[0]:
        raise ValueError(
            f"Response dimension {n_neurons} doesn't match "
            f"diffusion config conditioning.mlp_layers[0]={cfg.conditioning.mlp_layers[0]} "
            f"(use DiffusionConfig.for_encoder(encoder))"
        )
    dataset = TinyImageNetWithResponses(image_dataset, responses)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    # ── Build model + scheduler ──────────────────────────────────────────
    model = build_diffusion_model(cfg).to(device)
    scheduler = build_scheduler(cfg)
    n_train, n_total = count_parameters(model)
    say(f"[Stage 3] Model parameters: {n_train:,} trainable / {n_total:,} total")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
    )

    steps_per_epoch = len(loader)
    total_steps = cfg.train.epochs * steps_per_epoch
    warmup_steps = cfg.train.warmup_steps

    use_bf16 = cfg.train.precision == "bfloat16" and device.type == "cuda"
    # bfloat16 only on a GPU; otherwise no autocast at all (older torch versions refuse a float32 autocast on the CPU)
    autocast = (lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16)) if use_bf16 else contextlib.nullcontext
    grad_clip = float(cfg.train.gradient_clip)
    sigma = float(cfg.conditioning.observation_noise_sigma)
    p_drop = float(cfg.conditioning.cfg_dropout_rate)

    history = {"epoch": [], "loss": []}
    start_epoch, global_step = 1, 0

    # last.pt holds everything needed to continue an interrupted run: model,
    # optimizer, step counter, loss history and the random-number state at the
    # end of the last completed epoch (so data order and noise draws continue
    # as if the run had not stopped). It is rewritten after every epoch and
    # deleted when the run completes.
    last_path = out_dir / "last.pt"
    if resume and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch, global_step = state["epoch"] + 1, state["global_step"]
        history = state["history"]
        torch.set_rng_state(state["rng_cpu"])
        if device.type == "cuda" and state.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(state["rng_cuda"], device)
        say(f"[Stage 3] Resuming from {last_path} after epoch {state['epoch']}")
    say(f"[Stage 3] Training for {cfg.train.epochs} epochs "
        f"({total_steps} steps, batch={cfg.train.batch_size}) ...")

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        pbar = tqdm(loader, desc=f"epoch {epoch:3d}/{cfg.train.epochs}", leave=False, disable=not progress)
        for x, r in pbar:
            x = x.to(device, non_blocking=True)                 # [B, 1, 64, 64] in [-1, 1]
            r = r.to(device, non_blocking=True)                 # [B, n_neurons]

            B = x.shape[0]

            # 1) Forward diffuse: x_t = √ᾱ_t x + √(1−ᾱ_t) z
            t = torch.randint(0, cfg.scheduler.num_train_timesteps, (B,), device=device)
            z = torch.randn_like(x)
            x_t = scheduler.add_noise(x, z, t)

            # 2) Inject observation noise into conditioning: r̃ = r + σ ξ
            r_tilde = r + sigma * torch.randn_like(r)

            # 3) CFG dropout: replace r̃ by zeros for a fraction p_drop of samples
            if p_drop > 0:
                drop_mask = (torch.rand(B, device=device) < p_drop).view(B, 1)
                r_tilde = torch.where(drop_mask, torch.zeros_like(r_tilde), r_tilde)

            # 4) Predict + MSE on noise
            with autocast():
                eps_pred = model(x_t, t, r_tilde)
                loss = F.mse_loss(eps_pred, z)

            # 5) Backward + clip + step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # Cosine LR schedule with warmup, applied per-step.
            lr_factor = cosine_with_warmup(global_step, warmup_steps, total_steps)
            for g in optimizer.param_groups:
                g["lr"] = cfg.train.lr * lr_factor

            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{cfg.train.lr * lr_factor:.2e}")

        epoch_loss /= max(1, n_batches)
        history["epoch"].append(epoch)
        history["loss"].append(epoch_loss)
        say(f"  epoch {epoch:3d}/{cfg.train.epochs}: mean loss = {epoch_loss:.4f}")

        # Overwrite the latest model every epoch (so a crash loses at most one
        # epoch) and keep a periodic snapshot. After the last epoch,
        # final_model.pt is the checkpoint used for every figure.
        ckpt = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "loss": epoch_loss,
            "provenance": prov,
            **metadata,
        }
        torch.save(ckpt, out_dir / "final_model.pt")
        if epoch % save_every == 0:
            torch.save(ckpt, out_dir / f"model_epoch_{epoch:03d}.pt")
        with open(out_dir / "train_log.json", "w") as f:
            json.dump(history, f, indent=2)
        tmp = out_dir / "last.pt.tmp"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "history": history,
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }, tmp)
        tmp.replace(last_path)                     # atomic: never a half-written last.pt

    last_path.unlink(missing_ok=True)
    say(f"[Stage 3] Saved checkpoint to {out_dir / 'final_model.pt'} (epoch {cfg.train.epochs})")
    return history


def main(argv=None):
    """Train the diffusion model from a responses file (command-line entry point)."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.data_dir:
        os.environ["NEURIPS_DATA_DIR"] = args.data_dir

    device = get_device(args.device)

    # ── Load responses produced by Stage 2 ───────────────────────────────
    print(f"[Stage 3] Loading responses from {args.responses} ...")
    resp_pkg = torch.load(args.responses, map_location="cpu", weights_only=False)
    responses = resp_pkg["responses"]                    # [N, n_neurons]
    print(f"  responses: {tuple(responses.shape)}, encoder='{resp_pkg.get('encoder_name', '?')}', "
          f"area='{resp_pkg.get('area', '?')}'")

    # ── Pair the images with the responses ───────────────────────────────
    image_dataset = TinyImageNetDataset(split=cfg.train.data_split, transform=CANONICAL_TRANSFORM)
    if len(image_dataset) > cfg.train.n_train_images:
        # Truncate deterministically (sorted order from the data loader).
        image_dataset.image_paths = image_dataset.image_paths[: cfg.train.n_train_images]
        responses = responses[: cfg.train.n_train_images]
    print(f"  paired dataset size: {len(image_dataset)}")

    train_diffusion(
        cfg, image_dataset, responses, args.out_dir,
        device=device, num_workers=args.num_workers, save_every=args.save_every, resume=args.resume,
        metadata={
            "config": {"diffusion": str(args.config), "responses_path": str(args.responses)},
            "encoder_name": resp_pkg.get("encoder_name"),
            "area": resp_pkg.get("area"),
        },
        prov=provenance({"diffusion": args.config}, seed=cfg.train.seed),
    )


if __name__ == "__main__":
    main()
