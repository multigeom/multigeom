"""
Stage 2: Generate synthetic neural responses for every TinyImageNet image
by evaluating the trained encoder f_φ from Stage 1. These vectors form the
conditioning input for the diffusion model trained in Stage 3.

The TinyImageNet image distribution is the diffusion-training distribution,
chosen because the THINGS dataset is too small to train a diffusion model
on directly. Conditioning on synthetic f_φ(x) responses approximates the
joint p(x, r) the multi-scale Fisher integrand integrates over.

Output: train_responses.pt and val_responses.pt under --out-dir, each a
torch.Tensor of shape [n_images, n_neurons].

Usage:
    python -m multigeom.diffusion.responses \\
        --config inception --area V1 \\
        --encoder-ckpt runs/encoders/inception/V1/best_model.pt \\
        --out-dir runs/responses/inception/V1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


from ..config import load_config
from ..datasets import CANONICAL_TRANSFORM, TinyImageNetDataset
from ..encoders import build_encoder
from ..utils import get_device, provenance


def parse_args(argv=None):
    """Command-line arguments of `python -m multigeom.diffusion.responses`."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="encoder config YAML")
    p.add_argument("--area", required=True, choices=["V1", "V4"])
    p.add_argument(
        "--encoder-ckpt",
        required=True,
        help="Stage-1 checkpoint produced by 01_train_encoder.py",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    return p.parse_args(argv)


@torch.no_grad()
def precompute_responses(
    encoder, dataset, batch_size: int = 256, device=None, num_workers: int = 0, log=None
) -> torch.Tensor:
    """
    The responses [len(dataset), n_neurons] of `encoder` to every image of `dataset`
    (images [1, 64, 64] in [-1, 1]; the items may also be (image, label) pairs): the
    conditioning input of the diffusion model. Works with any encoder that follows the
    contract of `multigeom.encoders.contract`.
    """
    if device is not None:
        encoder.to(device)
    device = next(encoder.parameters()).device
    was_training = encoder.training
    encoder.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    responses = []
    for batch_idx, batch in enumerate(loader):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        responses.append(encoder(x.to(device, non_blocking=True)).cpu())
        if log and (batch_idx + 1) % 50 == 0:
            done = min((batch_idx + 1) * batch_size, len(dataset))
            log(f"    {done:6d}/{len(dataset)}")
    encoder.train(was_training)
    return torch.cat(responses, dim=0)


def precompute_split(encoder, split: str, transform, device, batch_size: int):
    """Responses of `encoder` to every image of a TinyImageNet split."""
    dataset = TinyImageNetDataset(split=split, transform=transform)
    print(f"  {split}: {len(dataset)} images")
    return precompute_responses(
        encoder, dataset, batch_size=batch_size, device=device, num_workers=4, log=print
    )


def main(argv=None):
    """
    Precompute and save the responses of an encoder on TinyImageNet (command-line entry
    point).
    """
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.data_dir:
        os.environ["NEURIPS_DATA_DIR"] = args.data_dir

    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build encoder + load Stage-1 checkpoint ──
    print(f"[Stage 2] Building {cfg.encoder.name} encoder ...")
    encoder = build_encoder(cfg).to(device)
    ckpt = torch.load(args.encoder_ckpt, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder.eval()
    print(
        f"  Loaded checkpoint (test_corr = {ckpt.get('test_corr', float('nan')):.3f})"
    )

    # ── Generate responses for each requested split ──
    # Responses are computed on exactly the [-1, 1] images the diffusion model
    # is trained on (Stage 3); every encoder converts to its backbone's range
    # internally.
    for split in args.splits:
        print(
            f"[Stage 2] Generating synthetic responses for TinyImageNet '{split}' ..."
        )
        responses = precompute_split(
            encoder, split, CANONICAL_TRANSFORM, device, args.batch_size
        )
        out_path = out_dir / f"{split}_responses.pt"
        torch.save(
            {
                "responses": responses,
                "n_neurons": responses.shape[1],
                "area": args.area,
                "encoder_name": cfg.encoder.name,
                "encoder_ckpt": str(args.encoder_ckpt),
                "provenance": provenance({"encoder": args.config}),
            },
            out_path,
        )
        print(f"  saved {tuple(responses.shape)} → {out_path}")

    print("[Stage 2] Done.")


if __name__ == "__main__":
    main()
