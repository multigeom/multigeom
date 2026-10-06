"""
Stage 4: For each of the held-out test images, compute

  - the top-k eigenpairs of the multi-scale Fisher metric G(x) via the
    **conditional-pair estimator** (Algorithm 1 of the paper, Eq. (eq:cc));
  - the top-k eigenpairs of the closed-form Fisher baseline J(x) =
    (1/σ²) J_{f_φ}ᵀ J_{f_φ} via PyTorch reverse-mode autograd (App. C.2).

Outputs one file per image: `<out_dir>/img{N:04d}.pt` with both
eigendecompositions and metadata. Stage 5 reads these to produce the
cross-encoder figures.

Test images: the first `n_test_images` images of the TinyImageNet validation
split (held out from diffusion training), in sorted file order. The set is
deterministic and identical for every (encoder, area) configuration.

Seeding: image i uses its own generator seeded with `seed + i`, so its
estimate does not depend on how images are split across parallel runs
(--start/--end).

Usage:
    python stages/04_estimate_gx.py \\
        --config configs/estimator.yaml \\
        --encoder-cfg configs/inception.yaml \\
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loaders import TinyImageNetDataset, CANONICAL_TRANSFORM
from src.diffusion_model import build_diffusion_model, build_scheduler
from src.encoders import build_encoder
from src.estimator import conditional_pair_estimate
from src.fisher import fisher_eigendecomposition
from src.utils import get_device, load_config, make_generator, provenance, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="estimator config YAML (configs/estimator.yaml)")
    p.add_argument("--encoder-cfg", required=True,
                   help="encoder config YAML matching the trained checkpoint")
    p.add_argument("--encoder-ckpt", required=True)
    p.add_argument("--diffusion-ckpt", required=True)
    p.add_argument("--diffusion-cfg", default=None,
                   help="diffusion config YAML (default: configs/diffusion.yaml)")
    p.add_argument("--area", required=True, choices=["V1", "V4"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--start", type=int, default=0,
                   help="start index in the test image list (for parallel runs)")
    p.add_argument("--end", type=int, default=None,
                   help="end index (exclusive). Default: cfg.n_test_images.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    diff_cfg_path = args.diffusion_cfg or str(
        Path(__file__).resolve().parent.parent / "configs" / "diffusion.yaml"
    )
    diff_cfg = load_config(diff_cfg_path)
    enc_cfg = load_config(args.encoder_cfg)

    if args.data_dir:
        os.environ["NEURIPS_DATA_DIR"] = args.data_dir

    device = get_device(args.device)
    set_seed(cfg.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load encoder f_φ ─────────────────────────────────────────────────
    print(f"[Stage 4] Loading encoder ({enc_cfg.encoder.name}, {args.area}) ...")
    encoder = build_encoder(enc_cfg).to(device)
    enc_ckpt = torch.load(args.encoder_ckpt, map_location=device, weights_only=False)
    encoder.load_state_dict(enc_ckpt["model_state_dict"])
    encoder.eval()
    print(f"  test_corr (Stage 1): {enc_ckpt.get('test_corr', float('nan')):.3f}")

    # ── Load diffusion model ε_θ + scheduler ─────────────────────────────
    print(f"[Stage 4] Loading diffusion model ...")
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
    print(f"[Stage 4] Test images: {args.start}–{end - 1} of first {n_test} val images.")

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
    fisher_top_k = int(cfg.fisher_baseline.top_k) if cfg.fisher_baseline.enabled else 0

    # ── Per-image loop ───────────────────────────────────────────────────
    for img_idx in tqdm(range(args.start, end), desc=f"{enc_cfg.encoder.name}/{args.area}"):
        out_path = out_dir / f"img{img_idx:04d}.pt"
        if out_path.exists():
            continue

        x = test_dataset[img_idx].unsqueeze(0)            # [1, 1, 64, 64] in [-1, 1]
        x = x.to(device)

        t0 = time.time()

        # Conditional-pair estimator (Algorithm 1) for G(x)
        g_out = conditional_pair_estimate(
            image=x,
            encoder=encoder,
            unet=unet,
            alphas_cumprod=alphas_cumprod,
            cfg=cfg,
            device=device,
            generator=make_generator(int(cfg.seed) + img_idx, device),
        )

        # Closed-form Fisher baseline J(x)
        if cfg.fisher_baseline.enabled:
            j_out = fisher_eigendecomposition(
                image=x,
                encoder=encoder,
                sigma=sigma,
                top_k=fisher_top_k,
                device=device,
            )
        else:
            j_out = None

        elapsed = time.time() - t0

        torch.save({
            "img_idx": img_idx,
            "image": x.cpu(),                              # [1, 1, 64, 64] in [-1, 1]
            "encoder_name": enc_cfg.encoder.name,
            "area": args.area,
            "G_eigenvalues": g_out.eigenvalues.cpu(),
            "G_eigenvectors": g_out.eigenvectors.cpu(),
            "J_eigenvalues": j_out.eigenvalues.cpu() if j_out is not None else None,
            "J_eigenvectors": j_out.eigenvectors.cpu() if j_out is not None else None,
            "estimator": "conditional_pair",
            "sigma": sigma,
            "elapsed_seconds": elapsed,
            "diffusion_epoch": diff_ckpt.get("epoch"),
            "provenance": prov,
        }, out_path)

    print(f"[Stage 4] Done. Per-image outputs in {out_dir}")


if __name__ == "__main__":
    main()
