"""
Truncation-layer sweep for the frozen-backbone encoders (paper App.
"InceptionV3 model"):

    "We chose the truncation layer empirically, by sweeping every candidate
     Inception block against each area under a fixed readout architecture and
     training schedule. Conv2d_4a_3x3 maximised test correlation in both V1
     and V4; deeper blocks did not help V4 and hurt V1."

The same sweep is available for the ResNet-50 backbone (blocks conv1, maxpool,
layer1.0 ... layer4.2), whose chosen truncation is layer1.0.

For every candidate layer ℓ the encoder is

    r̂_n(x) = Σ_c w_{n,c} Σ_{h,w} a_n(h,w) φ^ℓ_c(x)_{h,w} + b_n     (Eq. (eq:readout))

with φ^ℓ the frozen backbone truncated after ℓ and a_n the normalised Gaussian
attention of `multigeom.encoders.GaussianReadout`. Everything except ℓ is held
fixed: the canonical [-1, 1] 64×64 grayscale input, the backbone's input
conversion (`multigeom.encoders._BaseEncoder._to_backbone_input`), the readout, and
the Stage-1 recipe read from the encoder config (AdamW, cosine schedule, MSE,
batch size, gradient clipping at 1.0, 5% image-level validation split with a
fixed seed, model selection on mean validation correlation). The reported
score is the mean Pearson correlation over the n target neurons on the Papale
test set (and its noise-corrected version r_n/ρ_n).

Truncating at an arbitrary layer. `multigeom.encoders.InceptionBackbone` keeps the
torchvision children up to the named layer, which is correct up to Mixed_6e
but would include the auxiliary classifier (`AuxLogits`) for deeper layers,
and `multigeom.encoders.ResNet50Backbone` accepts only "<stage>.<block>" names.
`TruncatedBackbone` in this file walks the same child list, skips `AuxLogits`
and expands each ResNet stage into its blocks, so `layer1.0` and
`Conv2d_4a_3x3` give exactly the modules of the multigeom backbones. Layers whose
output is empty for a 64×64 input (Inception Mixed_7a onwards, after the
feature map has shrunk to 2×2) are recorded as skipped.

Features are extracted for one layer at a time, from images decoded once and
kept in memory, so peak memory is that of the largest single layer.

Output (JSON): per layer the feature shape, best validation correlation, test
correlation (raw and noise-corrected), best epoch and wall time; the best
layer; and a provenance record.

Usage:
    python -m multigeom.fit.layer_sweep --backbone inception --area V1 \\
        --out runs/analysis/layer_sweep_inception_V1.json
    python -m multigeom.fit.layer_sweep --backbone resnet50 --area V4 --layers layer1.0 layer2.0
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


from ..config import config_path, data_dir, load_config
from ..encoders import InceptionEncoder, ResNet50Backbone, ResNet50Encoder, _BaseEncoder, _FrozenBackbone
from ..encoders.fitting import correlation_per_neuron, fit_readout_on_features
from ..utils import get_device, provenance, set_seed

INCEPTION_LAYERS = [
    "Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3", "Conv2d_3b_1x1", "Conv2d_4a_3x3",
    "Mixed_5b", "Mixed_5c", "Mixed_5d",
    "Mixed_6a", "Mixed_6b", "Mixed_6c", "Mixed_6d", "Mixed_6e",
    "Mixed_7a", "Mixed_7b", "Mixed_7c",
]
RESNET_LAYERS = (
    ["conv1", "maxpool"]
    + [f"layer1.{i}" for i in range(3)] + [f"layer2.{i}" for i in range(4)]
    + [f"layer3.{i}" for i in range(6)] + [f"layer4.{i}" for i in range(3)]
)


# ────────────────────────────────────────────────────────────────────────────
# Backbone truncated at an arbitrary layer
# ────────────────────────────────────────────────────────────────────────────
def _flat_children(net: nn.Module, backbone: str) -> list[tuple[str, nn.Module]]:
    """Ordered (name, module) list up to the classifier, ResNet stages split into blocks."""
    out = []
    for name, mod in net.named_children():
        if name in ("AuxLogits", "avgpool", "dropout", "fc"):
            continue
        if backbone == "resnet50" and name.startswith("layer"):
            out.extend((f"{name}.{i}", block) for i, block in mod.named_children())
        else:
            out.append((name, mod))
    return out


class TruncatedBackbone(_FrozenBackbone):
    """
    Frozen torchvision Inception-V3 or ResNet-50 truncated after `layer`.

    The ResNet name "conv1" denotes conv1 → bn1 → relu (the stem before
    max-pooling). ResNet input is standardised with the ImageNet statistics,
    as in `multigeom.encoders.ResNet50Backbone`.
    """

    def __init__(self, backbone: str, layer: str, net: nn.Module | None = None,
                 spatial: int = 64):
        super().__init__()
        if net is None:
            net = build_torchvision_net(backbone, pretrained=True)
        children = _flat_children(net, backbone)
        stop = "relu" if (backbone == "resnet50" and layer == "conv1") else layer
        names = [n for n, _ in children]
        if stop not in names:
            raise ValueError(f"unknown {backbone} layer {layer!r}")
        self.features = nn.Sequential()
        for name, mod in children[: names.index(stop) + 1]:
            self.features.add_module(name.replace(".", "_"), mod)
        self.standardise = backbone == "resnet50"
        if self.standardise:
            self.register_buffer("norm_mean",
                                 torch.tensor(ResNet50Backbone.IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer("norm_std",
                                 torch.tensor(ResNet50Backbone.IMAGENET_STD).view(1, 3, 1, 1))
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        with torch.no_grad():
            shape = self.forward(torch.zeros(1, 3, spatial, spatial)).shape
        if shape[-1] != shape[-2]:
            raise ValueError(f"non-square feature map {tuple(shape)}")
        self.out_channels, self.out_spatial = int(shape[1]), int(shape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feature map of the truncated backbone."""
        if self.standardise:
            x = (x - self.norm_mean) / self.norm_std
        return self.features(x)


def build_torchvision_net(backbone: str, pretrained: bool) -> nn.Module:
    """A torchvision InceptionV3 or ResNet-50, with ImageNet weights if `pretrained`."""
    from torchvision.models import (
        Inception_V3_Weights, ResNet50_Weights, inception_v3, resnet50,
    )
    if backbone == "inception":
        w = Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        return inception_v3(weights=w, aux_logits=True, init_weights=not pretrained)
    if backbone == "resnet50":
        return resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    raise ValueError(f"unknown backbone {backbone!r}")


class _SweepEncoder(_BaseEncoder):
    """Truncated backbone + Gaussian readout."""

    def __init__(self, backbone: str, layer: str, n_neurons: int, dropout: float,
                 net: nn.Module | None = None):
        super().__init__(TruncatedBackbone(backbone, layer, net=net), n_neurons, dropout)


class InceptionSweepEncoder(_SweepEncoder):
    """Encoder on an InceptionV3 truncated at any layer."""
    backbone_input_channels = InceptionEncoder.backbone_input_channels
    backbone_input_range = InceptionEncoder.backbone_input_range


class ResNetSweepEncoder(_SweepEncoder):
    """Encoder on a ResNet-50 truncated at any block."""
    backbone_input_channels = ResNet50Encoder.backbone_input_channels
    backbone_input_range = ResNet50Encoder.backbone_input_range


def SweepEncoder(backbone: str, layer: str, n_neurons: int, dropout: float,
                 net: nn.Module | None = None) -> _SweepEncoder:
    """Sweep encoder with the input conversion of the matching multigeom encoder."""
    cls = InceptionSweepEncoder if backbone == "inception" else ResNetSweepEncoder
    return cls(backbone, layer, n_neurons, dropout, net=net)


# ────────────────────────────────────────────────────────────────────────────
# Readout training (Stage-1 recipe on cached features)
# ────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _predict(readout: nn.Module, feats: torch.Tensor, device, batch_size: int = 256):
    readout.eval()
    return torch.cat([readout(feats[i:i + batch_size].to(device)).cpu()
                      for i in range(0, len(feats), batch_size)])


@torch.no_grad()
def extract(encoder: _BaseEncoder, images: torch.Tensor, device, batch_size: int = 128):
    """Frozen-backbone features of in-memory canonical images, kept on the CPU."""
    encoder.to(device).eval()
    return torch.cat([encoder.features(images[i:i + batch_size].to(device)).cpu()
                      for i in range(0, len(images), batch_size)])


def sweep(layers: list[str], backbone: str, images: dict, responses: dict, oracle: np.ndarray,
          train_cfg, n_neurons: int, dropout: float, device, seed: int,
          pretrained: bool = True, log=print) -> dict:
    """
    Train one readout per layer and score it on the test split.

    Args:
        images:    {"train": [N_tr, 1, 64, 64], "test": [N_te, 1, 64, 64]} in [-1, 1]
        responses: {"train": [N_tr, n], "test": [N_te, n]}
        oracle:    [n] split-half reliabilities of the target neurons
    """
    corr_fn = correlation_per_neuron
    net = build_torchvision_net(backbone, pretrained)

    results = {}
    for layer in layers:
        t0 = time.time()
        set_seed(seed)
        try:
            enc = SweepEncoder(backbone, layer, n_neurons, dropout, net=net)
        except (RuntimeError, ValueError) as e:      # map collapsed below the kernel size
            results[layer] = {"skipped": f"{type(e).__name__}: {str(e).splitlines()[0]}"}
            log(f"  {layer:16s} skipped ({results[layer]['skipped'][:60]})")
            continue
        f_tr = extract(enc, images["train"], device)
        f_te = extract(enc, images["test"], device)
        # The one readout-fitting loop of the package; the cached features stay on the CPU, batches go to `device`.
        fit = fit_readout_on_features(
            enc.readout, f_tr, responses["train"], epochs=train_cfg.epochs, lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay, batch_size=train_cfg.batch_size,
            val_fraction=train_cfg.validation_fraction, seed=seed, device=device, log=None)
        val_corr, best_epoch = fit.best_val_corr, fit.best_epoch
        test_corrs = corr_fn(_predict(enc.readout, f_te, device), responses["test"])
        results[layer] = {
            "feature_shape": list(f_tr.shape[1:]),
            "readout_parameters": int(sum(p.numel() for p in enc.readout.parameters())),
            "val_corr": val_corr,
            "best_epoch": best_epoch,
            "test_corr": float(test_corrs.mean()),
            "test_corr_noise_corrected": float((test_corrs / oracle).mean()),
            "per_neuron_test_corr": test_corrs.tolist(),
            "seconds": time.time() - t0,
        }
        log(f"  {layer:16s} {str(list(f_tr.shape[1:])):16s} val r = {val_corr:.4f}  "
            f"test r = {results[layer]['test_corr']:.4f}  ({results[layer]['seconds']:.0f}s)")
        del f_tr, f_te, enc
    return results


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None):
    """Command-line arguments of `python -m multigeom.fit.layer_sweep`."""
    p = argparse.ArgumentParser(description="Backbone truncation-layer sweep")
    p.add_argument("--backbone", choices=["inception", "resnet50"], default="inception")
    p.add_argument("--area", required=True, choices=["V1", "V4"])
    p.add_argument("--encoder-cfg", default=None,
                   help="config whose encoder.n_neurons/dropout and train section are used "
                        "(default: the packaged <backbone> config)")
    p.add_argument("--layers", nargs="+", default=None, help="subset of layers to sweep")
    p.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    p.add_argument("--images-root", default=None,
                   help="THINGS object_images root (default: <data_dir>/things_images/object_images)")
    p.add_argument("--tvsd-dir", default=None)
    p.add_argument("--max-train-images", type=int, default=None, help="smoke test: subsample")
    p.add_argument("--random-weights", action="store_true",
                   help="smoke test: randomly initialised backbone (no download)")
    p.add_argument("--device", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def _load_split(paths, responses, images_root, max_n=None):
    from PIL import Image
    from ..datasets import CANONICAL_TRANSFORM
    from .tvsd import existing_image_mask
    keep = np.flatnonzero(existing_image_mask(paths, images_root))
    if max_n is not None:
        keep = keep[:max_n]
    if keep.size == 0:
        raise FileNotFoundError(f"No THINGS images found under {images_root}")
    imgs = torch.stack([CANONICAL_TRANSFORM(Image.open(Path(images_root) / paths[i]).convert("RGB"))
                        for i in keep])
    return imgs, responses[keep], int(len(paths) - keep.size)


def main(argv: list[str] | None = None) -> dict:
    """Fit and score an encoder at every candidate layer (command-line entry point)."""
    from .tvsd import load_tvsd

    args = parse_args(argv)
    if args.data_dir:
        os.environ["NEURIPS_DATA_DIR"] = args.data_dir
    cfg_path = args.encoder_cfg or str(config_path(args.backbone))
    cfg = load_config(cfg_path)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    device = get_device(args.device)
    set_seed(cfg.train.seed)
    layers = args.layers or (INCEPTION_LAYERS if args.backbone == "inception" else RESNET_LAYERS)

    tvsd = load_tvsd(args.area, n_neurons=cfg.encoder.n_neurons, tvsd_dir=args.tvsd_dir)
    root = Path(args.images_root) if args.images_root else (
        data_dir() / "things_images" / "object_images")
    images, responses, missing = {}, {}, {}
    for split in ("train", "test"):
        cap = args.max_train_images if split == "train" else None
        images[split], responses[split], missing[split] = _load_split(
            tvsd[f"{split}_paths"], tvsd[f"{split}_responses"], root, cap)
    print(f"[layer_sweep] {args.backbone}/{args.area}: {len(images['train'])} train, "
          f"{len(images['test'])} test images; {len(layers)} layers; "
          f"{cfg.train.epochs} epochs per layer")

    results = sweep(layers, args.backbone, images, responses, tvsd["oracle"], cfg.train,
                    cfg.encoder.n_neurons, cfg.encoder.dropout, device, cfg.train.seed,
                    pretrained=not args.random_weights)
    scored = {k: v for k, v in results.items() if "test_corr" in v}
    best = max(scored, key=lambda k: scored[k]["test_corr"]) if scored else None
    best_val = max(scored, key=lambda k: scored[k]["val_corr"]) if scored else None
    out = {
        "backbone": args.backbone,
        "area": args.area,
        "n_train_images": len(images["train"]),
        "n_test_images": len(images["test"]),
        "missing_images": missing,
        "random_weights": bool(args.random_weights),
        "train": vars(cfg.train),
        "results": results,
        "best_layer_by_test_corr": best,
        "best_layer_by_val_corr": best_val,
        "provenance": provenance({"encoder": cfg_path}, seed=cfg.train.seed, argv=argv),
    }
    print(f"[layer_sweep] best layer (test r): {best}   (val r): {best_val}")
    out_path = Path(args.out or f"runs/analysis/layer_sweep_{args.backbone}_{args.area}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[layer_sweep] saved {out_path}")
    return out


if __name__ == "__main__":
    main()
