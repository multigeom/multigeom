"""
Stage 1: Train the per-neuron Gaussian spatial-attention readout on top of a
frozen backbone (Inception, ResNet-50, or ON-OFF), targeting the 50 most
reliable channels of macaque V1 or V4 from the THINGS Ventral-stream Spiking
Dataset (Papale et al.).

Backbone weights are held fixed; only the ~300-13,000 readout parameters are
optimised. To keep training fast we pre-extract backbone features once over
the full TVSD train + test split, then fit the readout on the cached
[N, C, H, W] features with AdamW + cosine schedule.

With --evaluate-only, an existing encoder (--encoder-ckpt) is not trained: it
is run on the TVSD test images, and its test-set predictions and per-neuron
correlations (raw and noise-corrected) are added to a copy of the checkpoint
written to --out-dir. These are the fields the figure and table scripts read
(Fig. 3B-C, Fig. 4A, Table 1). Only the test images are needed.

Usage:
    python -m multigeom.fit.train_encoder \\
        --config inception --area V1 \\
        --device cuda:0 \\
        --out-dir runs/encoders/inception/V1

    python -m multigeom.fit.train_encoder \\
        --config inception --area V1 --evaluate-only \\
        --encoder-ckpt runs/encoders/inception/V1/best_model.pt \\
        --out-dir runs/encoders/inception/V1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from ..config import data_dir, load_config
from ..data import CANONICAL_TRANSFORM
from ..encoders import build_encoder
from ..encoders.fitting import correlation_per_neuron, fit_readout_on_features
from ..utils import count_parameters, get_device, provenance, set_seed
from .tvsd import TVSDDataset, existing_image_mask, load_tvsd


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="encoder config YAML")
    p.add_argument("--area", required=True, choices=["V1", "V4"])
    p.add_argument("--out-dir", required=True, help="checkpoint + log directory")
    p.add_argument("--device", default=None, help="torch device (default: auto)")
    p.add_argument("--data-dir", default=None, help="override NEURIPS_DATA_DIR")
    p.add_argument("--evaluate-only", action="store_true",
                   help="do not train: evaluate --encoder-ckpt on the TVSD test images and "
                        "write it, with the test-set fields added, to <out-dir>/best_model.pt")
    p.add_argument("--encoder-ckpt", default=None,
                   help="existing encoder checkpoint (with --evaluate-only)")
    args = p.parse_args()
    if args.evaluate_only and not args.encoder_ckpt:
        p.error("--evaluate-only needs --encoder-ckpt")
    return args


def test_set_fields(test_pred: torch.Tensor, test_resp: torch.Tensor,
                    oracle: np.ndarray) -> dict:
    """
    Test-set entries stored in an encoder checkpoint. The noise-corrected
    correlation of neuron n is r_n / ρ_n (Table 1), with ρ_n its split-half
    oracle reliability.
    """
    corrs = correlation_per_neuron(test_pred, test_resp)
    corrs_nc = corrs / oracle
    return {
        "test_corr": float(corrs.mean()),
        "test_corr_noise_corrected": float(corrs_nc.mean()),
        "per_neuron_test_corr": corrs,
        "per_neuron_test_corr_noise_corrected": corrs_nc,
        "test_predictions": test_pred.detach().cpu(),   # [n_test, n_neurons] (Fig. 3B-C)
        "test_responses": test_resp.detach().cpu(),     # [n_test, n_neurons]
    }


def print_test_summary(fields: dict) -> None:
    c = fields["per_neuron_test_corr"]
    print(f"[Stage 1] Test correlation: mean={c.mean():.3f}  median={np.median(c):.3f}  "
          f"range=[{c.min():.3f}, {c.max():.3f}]  "
          f"noise-corrected mean={fields['test_corr_noise_corrected']:.3f}")


@torch.no_grad()
def preextract(encoder, paths, responses, images_root, transform, device, batch_size: int = 64):
    """Run the frozen backbone over (paths, responses), returning cached features."""
    dataset = TVSDDataset(responses, paths, images_root, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    feats, resp = [], []
    for img, r in loader:
        img = img.to(device, non_blocking=True)
        feats.append(encoder.features(img).cpu())
        resp.append(r)
    return torch.cat(feats, dim=0), torch.cat(resp, dim=0)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.data_dir:
        os.environ["NEURIPS_DATA_DIR"] = args.data_dir

    device = get_device(args.device)
    set_seed(cfg.train.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load TVSD data + select top-50 channels by oracle reliability ──
    print(f"[Stage 1] Loading TVSD ({args.area}, top-{cfg.encoder.n_neurons}) ...")
    tvsd = load_tvsd(args.area, n_neurons=cfg.encoder.n_neurons)
    print(f"  Train: {len(tvsd['train_paths'])}   Test: {len(tvsd['test_paths'])}")
    print(f"  Mean oracle reliability across selected neurons: "
          f"{tvsd['oracle'].mean():.3f}")

    # ── 2. Build encoder ──
    print(f"[Stage 1] Building {cfg.encoder.name} encoder ...")
    encoder = build_encoder(cfg).to(device)
    encoder.eval()  # backbone is frozen; readout flips into train mode below
    n_train, n_total = count_parameters(encoder)
    print(f"  Parameters: {n_train:,} trainable / {n_total:,} total")

    if args.evaluate_only:
        ckpt = torch.load(args.encoder_ckpt, map_location="cpu", weights_only=False)
        encoder.load_state_dict(ckpt["model_state_dict"], strict=True)
        encoder.eval()

    # ── 3. Pre-extract frozen features (train + test; test only to evaluate) ──
    # Every encoder receives the canonical [-1, 1] image and converts it to
    # its backbone's range internally (multigeom/encoders).
    print("[Stage 1] Pre-extracting frozen-backbone features (one-time) ...")
    images_root = data_dir() / "things_images" / "object_images"
    splits = {}
    for split in (("test",) if args.evaluate_only else ("train", "test")):
        paths = tvsd[f"{split}_paths"]
        keep = existing_image_mask(paths, images_root)
        if not keep.any():
            raise FileNotFoundError(f"No THINGS images found under {images_root}")
        if not keep.all():
            print(f"  {split}: {int((~keep).sum())}/{len(paths)} images missing on disk, skipped")
        idx = np.flatnonzero(keep)
        splits[split] = preextract(
            encoder, [paths[i] for i in idx], tvsd[f"{split}_responses"][idx],
            images_root, CANONICAL_TRANSFORM, device,
        )
    test_feats, test_resp = splits["test"]

    if args.evaluate_only:
        with torch.no_grad():
            fields = test_set_fields(encoder.readout(test_feats.to(device)),
                                     test_resp, tvsd["oracle"])
        print_test_summary(fields)
        ckpt.update(fields)
        ckpt["neuron_indices"] = tvsd["neuron_indices"]
        ckpt["oracle"] = tvsd["oracle"]
        ckpt["evaluation_provenance"] = provenance({"encoder": args.config})
        ckpt_path = out_dir / "best_model.pt"
        torch.save(ckpt, ckpt_path)
        print(f"[Stage 1] Evaluated {args.encoder_ckpt}; saved to {ckpt_path}")
        return

    train_feats, train_resp = splits["train"]
    print(f"  Feature shape (train): {tuple(train_feats.shape)}")

    # Move cached features to GPU for fast iteration
    train_feats = train_feats.to(device)
    train_resp = train_resp.to(device)
    test_feats = test_feats.to(device)
    test_resp = test_resp.to(device)

    # ── 4. Fit the readout on the cached features (image-level train/val split, fixed seed) ──
    print(f"[Stage 1] Training readout for {cfg.train.epochs} epochs ...")
    report = fit_readout_on_features(
        encoder.readout, train_feats, train_resp,
        epochs=cfg.train.epochs, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
        batch_size=cfg.train.batch_size, val_fraction=cfg.train.validation_fraction, seed=cfg.train.seed,
    )
    history = report.history
    best_val_corr = report.best_val_corr
    best_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}

    # ── 7. Test on best checkpoint ──
    encoder.eval()
    with torch.no_grad():
        fields = test_set_fields(encoder.readout(test_feats), test_resp, tvsd["oracle"])
    print_test_summary(fields)

    # ── 8. Save checkpoint + training log ──
    ckpt_path = out_dir / "best_model.pt"
    torch.save({
        "model_state_dict": best_state,
        "config": {
            "encoder": vars(cfg.encoder),
            "train": vars(cfg.train),
            "area": args.area,
        },
        "best_epoch": int(np.argmax(history["val_corr"]) + 1),
        "val_corr": float(best_val_corr),
        **fields,
        "neuron_indices": tvsd["neuron_indices"],
        "oracle": tvsd["oracle"],
        "provenance": provenance({"encoder": args.config}, seed=cfg.train.seed),
    }, ckpt_path)
    with open(out_dir / "train_log.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Stage 1] Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
