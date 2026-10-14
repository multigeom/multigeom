"""
The paper's recipe for fitting an encoder to macaque recordings: load the 50 most reliable channels of V1 or V4 from the
THINGS Ventral-stream Spiking Dataset (Papale et al.), build a reference encoder (Inception, ResNet-50 or ON-OFF
backbone, Gaussian readout), fit its readout and score it on the test images.

This module holds only what is specific to that dataset. The readout is fitted by `NeuralEncoder.fit_readout`
(`multigeom.encoders.fitting`), the same method as for any other data, and the checkpoint is written by
`NeuralEncoder.save`: the weights and architecture, with the test scores (`per_neuron_test_corr`, `test_predictions`,
`oracle`, ...) in its metadata, which the figure and table code read (Fig. 3B-C, Fig. 4A, Table 1).

With --evaluate-only, an existing encoder (--encoder-ckpt) is not trained: it is run on the TVSD test images and written
to --out-dir with its test scores. Only the test images are needed.

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
from pathlib import Path

import numpy as np
import torch


from ..config import data_dir, load_config
from ..datasets import CANONICAL_TRANSFORM
from ..encoders import build_encoder
from ..encoders.fitting import correlation_per_neuron
from ..utils import count_parameters, get_device, provenance, set_seed
from .tvsd import existing_image_mask, load_images, load_tvsd


def parse_args(argv=None):
    """Command-line arguments of `python -m multigeom.fit.train_encoder`."""
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
    p.add_argument("--num-workers", type=int, default=4, help="worker processes that read the images")
    args = p.parse_args(argv)
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
    """Print the mean, median and range of the per-neuron test correlations."""
    c = fields["per_neuron_test_corr"]
    print(f"[Stage 1] Test correlation: mean={c.mean():.3f}  median={np.median(c):.3f}  "
          f"range=[{c.min():.3f}, {c.max():.3f}]  "
          f"noise-corrected mean={fields['test_corr_noise_corrected']:.3f}")


def main(argv=None):
    """Fit the readout of an encoder to the TVSD recordings, or evaluate a checkpoint (command-line entry point)."""
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
    encoder.eval()  # backbone is frozen; the fit switches the readout to training mode
    n_train, n_total = count_parameters(encoder)
    print(f"  Parameters: {n_train:,} trainable / {n_total:,} total")

    previous = {}
    if args.evaluate_only:
        encoder.load_checkpoint(args.encoder_ckpt)
        previous = checkpoint_scores(args.encoder_ckpt)

    # ── 3. Load the images of the splits (train + test; test only to evaluate) ──
    # Every encoder receives the canonical [-1, 1] image and converts it to its backbone's range internally.
    print("[Stage 1] Loading the THINGS images ...")
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
        splits[split] = (load_images([paths[i] for i in idx], images_root, CANONICAL_TRANSFORM,
                                     num_workers=args.num_workers),
                         tvsd[f"{split}_responses"][idx])
    test_images, test_resp = splits["test"]

    # ── 4. Fit the readout: the same method as for any other data ──
    history = None
    if not args.evaluate_only:
        train_images, train_resp = splits["train"]
        print(f"[Stage 1] Training readout for {cfg.train.epochs} epochs on {len(train_images)} images ...")
        report = encoder.fit_readout(
            train_images, train_resp, epochs=cfg.train.epochs, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
            batch_size=cfg.train.batch_size, val_fraction=cfg.train.validation_fraction, seed=cfg.train.seed,
            device=device)
        history = report.history
        previous = {"best_epoch": report.best_epoch, "val_corr": float(report.best_val_corr)}

    # ── 5. Test the (best) readout ──
    fields = test_set_fields(encoder.predict(test_images, device=device), test_resp, tvsd["oracle"])
    print_test_summary(fields)

    # ── 6. Save in the layout of `encoder.save`: the scores are its metadata ──
    ckpt_path = out_dir / "best_model.pt"
    metadata = {
        "area": args.area,
        "encoder": cfg.encoder.name,
        **previous,
        **fields,
        "neuron_indices": tvsd["neuron_indices"],
        "oracle": tvsd["oracle"],
    }
    if args.evaluate_only:
        metadata["evaluation_provenance"] = provenance({"encoder": args.config})
    else:
        metadata["train"] = vars(cfg.train)
        metadata["provenance"] = provenance({"encoder": args.config}, seed=cfg.train.seed)
        with open(out_dir / "train_log.json", "w") as f:
            json.dump(history, f, indent=2)
    encoder.save(ckpt_path, **metadata)
    print(f"[Stage 1] Saved checkpoint to {ckpt_path}")


def checkpoint_scores(path) -> dict:
    """The fit scores an existing checkpoint carries (`val_corr`, `best_epoch`), in either checkpoint layout."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    scores = {**ckpt, **ckpt.get("metadata", {})}
    return {k: scores[k] for k in ("best_epoch", "val_corr") if k in scores}


if __name__ == "__main__":
    main()
