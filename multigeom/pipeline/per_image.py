"""
Estimate G(x) and J(x) for each of the held-out test images (the former stage 4): for each image, compute

  - the top-k eigenpairs of the multi-scale Fisher metric G(x) via the
    **conditional-pair estimator** (Algorithm 1 of the paper, Eq. (eq:cc));
  - the top-k eigenpairs of the closed-form Fisher baseline J(x) =
    (1/σ²) J_{f_φ}ᵀ J_{f_φ} via PyTorch reverse-mode autograd (App. C.2).

Outputs one file per image: `<out_dir>/img{N:04d}.pt` with both
eigendecompositions and metadata. `multigeom.pipeline.alignment` reads these to produce the
cross-encoder figures.

Test images: the first `n_test_images` images of the TinyImageNet validation
split (held out from diffusion training), in sorted file order. The set is
deterministic and identical for every (encoder, area) configuration.

Seeding: image i uses its own generator seeded with `seed + i`, so its
estimate does not depend on how images are split across parallel runs
(--start/--end).

Usage:
    python -m multigeom.pipeline.per_image \\
        --config estimator \\
        --encoder-cfg inception \\
        --encoder-ckpt runs/encoders/inception/V1/best_model.pt \\
        --diffusion-ckpt runs/diffusion/inception/V1/final_model.pt \\
        --area V1 \\
        --out-dir runs/per_image/inception/V1
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm


from ..config import load_config
from ..datasets import CANONICAL_TRANSFORM, TinyImageNetDataset
from ..diffusion.model import build_diffusion_model, build_scheduler
from ..encoders import build_encoder
from ..geometry.estimator import conditional_pair_estimate
from ..geometry.fisher import fisher_eigendecomposition
from ..utils import get_device, make_generator, provenance, set_seed


def estimate_image(x, encoder, unet, alphas_cumprod, cfg, *, seed: int, device):
    """
    G(x) and J(x) of one image `x` ([1, 1, 64, 64] in [-1, 1]): the conditional-pair estimator (Algorithm 1) for G(x),
    with its own generator seeded with `seed` (so the result does not depend on how images are split over runs), and the
    closed-form Fisher information J(x). Returns (G output, J output or None if `cfg.fisher_baseline.enabled` is off).
    """
    x = x.to(device)
    sigma = float(cfg.monte_carlo.observation_noise_sigma)
    g_out = conditional_pair_estimate(
        image=x, encoder=encoder, unet=unet, alphas_cumprod=alphas_cumprod, cfg=cfg, device=device,
        generator=make_generator(int(seed), device),
    )
    j_out = None
    if cfg.fisher_baseline.enabled:
        j_out = fisher_eigendecomposition(image=x, encoder=encoder, sigma=sigma,
                                          top_k=int(cfg.fisher_baseline.top_k), device=device)
    return g_out, j_out


def estimate_images(encoder, unet, alphas_cumprod, images, indices, cfg, out_dir, *, encoder_name: str, area: str,
                    device, diffusion_epoch=None, provenance_record=None, desc: str | None = None,
                    skip_existing: bool = True) -> list[Path]:
    """
    Estimate G(x) and J(x) for `images[i]` (tensors [1, 64, 64] in [-1, 1]) for each i in `indices` and write one file
    `<out_dir>/img{i:04d}.pt` per image: the image, the top eigenvalues and eigenvectors of G(x) and J(x), the
    settings and the provenance. Image i is seeded with `cfg.seed + i`. Resumable: files that exist are skipped
    (`skip_existing`). Returns the files written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sigma = float(cfg.monte_carlo.observation_noise_sigma)
    written = []
    for img_idx in tqdm(list(indices), desc=desc or f"{encoder_name}/{area}"):
        out_path = out_dir / f"img{img_idx:04d}.pt"
        if skip_existing and out_path.exists():
            continue
        x = images[img_idx].unsqueeze(0)                  # [1, 1, 64, 64] in [-1, 1]
        t0 = time.time()
        g_out, j_out = estimate_image(x, encoder, unet, alphas_cumprod, cfg, seed=int(cfg.seed) + img_idx, device=device)
        elapsed = time.time() - t0
        torch.save({
            "img_idx": img_idx,
            "image": x.cpu(),                              # [1, 1, 64, 64] in [-1, 1]
            "encoder_name": encoder_name,
            "area": area,
            "G_eigenvalues": g_out.eigenvalues.cpu(),
            "G_eigenvectors": g_out.eigenvectors.cpu(),
            "J_eigenvalues": j_out.eigenvalues.cpu() if j_out is not None else None,
            "J_eigenvectors": j_out.eigenvectors.cpu() if j_out is not None else None,
            "estimator": "conditional_pair",
            "sigma": sigma,
            "elapsed_seconds": elapsed,
            "diffusion_epoch": diffusion_epoch,
            "provenance": provenance_record,
        }, out_path)
        written.append(out_path)
    return written


def parse_args(argv=None):
    """Command-line arguments of `python -m multigeom.pipeline.per_image`."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="estimator config: a packaged name ('estimator') or a YAML path")
    p.add_argument("--encoder-cfg", required=True,
                   help="encoder config YAML matching the trained checkpoint")
    p.add_argument("--encoder-ckpt", required=True)
    p.add_argument("--diffusion-ckpt", required=True)
    p.add_argument("--diffusion-cfg", default=None,
                   help="diffusion config: a packaged name or a YAML path (default: the packaged 'diffusion')")
    p.add_argument("--area", required=True, choices=["V1", "V4"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--start", type=int, default=0,
                   help="start index in the test image list (for parallel runs)")
    p.add_argument("--end", type=int, default=None,
                   help="end index (exclusive). Default: cfg.n_test_images.")
    return p.parse_args(argv)


def main(argv=None):
    """Estimate G(x) and J(x) for the test images (command-line entry point)."""
    args = parse_args(argv)
    cfg = load_config(args.config)

    diff_cfg_path = args.diffusion_cfg or "diffusion"
    diff_cfg = load_config(diff_cfg_path)
    enc_cfg = load_config(args.encoder_cfg)

    if args.data_dir:
        os.environ["NEURIPS_DATA_DIR"] = args.data_dir

    device = get_device(args.device)
    set_seed(cfg.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load encoder f_φ ─────────────────────────────────────────────────
    print(f"[G(x) estimation] Loading encoder ({enc_cfg.encoder.name}, {args.area}) ...")
    encoder = build_encoder(enc_cfg).to(device)
    enc_ckpt = torch.load(args.encoder_ckpt, map_location=device, weights_only=False)
    encoder.load_state_dict(enc_ckpt["model_state_dict"])
    encoder.eval()
    print(f"  test_corr (Stage 1): {enc_ckpt.get('test_corr', float('nan')):.3f}")

    # ── Load diffusion model ε_θ + scheduler ─────────────────────────────
    print(f"[G(x) estimation] Loading diffusion model ...")
    unet = build_diffusion_model(diff_cfg).to(device)
    diff_ckpt = torch.load(args.diffusion_ckpt, map_location=device, weights_only=False)
    unet.load_state_dict(diff_ckpt["model_state_dict"])
    unet.eval()
    scheduler = build_scheduler(diff_cfg)
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    # ── Load deterministic test images ───────────────────────────────────
    n_test = int(cfg.n_test_images)
    test_dataset = TinyImageNetDataset(split="val", transform=CANONICAL_TRANSFORM)
    end = args.end if args.end is not None else n_test
    if not (0 <= args.start < end <= n_test):
        raise ValueError(f"Invalid index range start={args.start}, end={end}, n_test={n_test}")
    if len(test_dataset) < end:
        raise ValueError(f"TinyImageNet val has {len(test_dataset)} images, need {end}")
    print(f"[G(x) estimation] Test images: {args.start}–{end - 1} of first {n_test} val images.")

    sigma = float(cfg.monte_carlo.observation_noise_sigma)
    sigma_train = float(diff_cfg.conditioning.observation_noise_sigma)
    if sigma != sigma_train:
        raise ValueError(
            f"Estimator σ={sigma} differs from the σ={sigma_train} the diffusion model "
            "was trained with; the observation model must be the same in both."
        )
    prov = provenance(
        {"estimator": args.config, "encoder": args.encoder_cfg, "diffusion": diff_cfg_path},
        seed=int(cfg.seed),
    )
    # ── Per-image loop ───────────────────────────────────────────────────
    estimate_images(
        encoder, unet, alphas_cumprod, test_dataset, range(args.start, end), cfg, out_dir,
        encoder_name=enc_cfg.encoder.name, area=args.area, device=device,
        diffusion_epoch=diff_ckpt.get("epoch"), provenance_record=prov,
        desc=f"{enc_cfg.encoder.name}/{args.area}",
    )

    print(f"[G(x) estimation] Done. Per-image outputs in {out_dir}")


if __name__ == "__main__":
    main()
