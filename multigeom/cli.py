"""
Command line: `multigeom <command> [options]` (or `python -m multigeom.cli`). Each command hands its options to the
module that does the work, so `multigeom <command> --help` lists them.

    jacobian         the Fisher metric J(x) of an encoder for one image or the first N test images
    estimate-g       the multi-scale metric G(x) (and J(x)) for the test images, resumable
    train-diffusion  the conditional diffusion model on precomputed responses
    align            subspace alignment and Bures-Wasserstein distance between configurations
    train-encoder    fit an encoder readout to neural data              (needs the [paper] extra)
    responses        precompute the encoder responses on TinyImageNet    (core)
    metric-field     the local metric field of Fig. 5
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

# command -> (module with a main(argv), one-line help)
COMMANDS = {
    "estimate-g": ("multigeom.pipeline.per_image", "estimate G(x) and J(x) for the test images (resumable)"),
    "train-diffusion": ("multigeom.diffusion.train", "train the conditional diffusion model"),
    "align": ("multigeom.pipeline.alignment", "alignment and Bures-Wasserstein distances between configurations"),
    "train-encoder": ("multigeom.fit.train_encoder", "fit an encoder readout to neural data"),
    "responses": ("multigeom.diffusion.responses", "precompute encoder responses on TinyImageNet"),
    "metric-field": ("multigeom.pipeline.metric_field", "local metric field (paper Fig. 5)"),
}


def jacobian(argv: list[str] | None = None) -> list[Path]:
    """J(x) = J_f^T J_f / sigma^2 of an encoder: its top eigenpairs for one image, or the first N test images."""
    p = argparse.ArgumentParser(prog="multigeom jacobian", description=jacobian.__doc__)
    p.add_argument("--encoder-cfg", required=True, help="encoder config: a packaged name ('inception') or a YAML path")
    p.add_argument("--encoder-ckpt", required=True)
    p.add_argument("--image", default=None, help="an image file; default: the first --n-images test images")
    p.add_argument("--n-images", type=int, default=1)
    p.add_argument("--sigma", type=float, default=0.22, help="observation noise of the encoder's responses")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)

    import torch
    from PIL import Image

    from .config import load_config
    from .data import CANONICAL_TRANSFORM, TinyImageNetDataset
    from .encoders import build_encoder
    from .geometry import fisher_eigendecomposition
    from .utils import get_device

    device = get_device(args.device)
    encoder = build_encoder(load_config(args.encoder_cfg)).to(device)
    encoder.load_state_dict(torch.load(args.encoder_ckpt, map_location=device, weights_only=False)["model_state_dict"])
    encoder.eval()
    if args.image:
        images = [("image", CANONICAL_TRANSFORM(Image.open(args.image).convert("RGB")))]
    else:
        data = TinyImageNetDataset(split="val", transform=CANONICAL_TRANSFORM)
        images = [(f"img{i:04d}", data[i])
                  for i in range(args.n_images)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, x in images:
        out = fisher_eigendecomposition(image=x.unsqueeze(0).to(device), encoder=encoder, sigma=args.sigma,
                                        top_k=args.top_k, device=device)
        path = out_dir / f"{name}.pt"
        torch.save({"J_eigenvalues": out.eigenvalues.cpu(), "J_eigenvectors": out.eigenvectors.cpu(),
                    "sigma": args.sigma}, path)
        written.append(path)
        print(path)
    return written


def main(argv: list[str] | None = None) -> int:
    """Run `multigeom <command>`; returns the exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: multigeom <command> [options]\n\ncommands:")
        print(f"  {'jacobian':16s} the Fisher metric J(x) of an encoder")
        for name, (_, text) in COMMANDS.items():
            print(f"  {name:16s} {text}")
        return 0
    command, rest = argv[0], argv[1:]
    if command == "jacobian":
        jacobian(rest)
        return 0
    if command not in COMMANDS:
        print(f"multigeom: unknown command {command!r} (see multigeom --help)", file=sys.stderr)
        return 2
    importlib.import_module(COMMANDS[command][0]).main(rest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
