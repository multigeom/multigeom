"""
Stage 6: Local metric field around one reference image (paper Fig. 5).

For a reference image x₀ (index into the same TinyImageNet validation list as
Stage 4), each area, and each metric M ∈ {G, J}:

  1. Compute the top eigenpairs of M(x₀) for the reference encoder
     (Inception by default) and keep the leading two, (λ₁, v₁), (λ₂, v₂).
     Panel A of Fig. 5 uses the G eigenvectors, panel B the J eigenvectors,
     both of the reference encoder.
  2. Build the n × n grid of perturbed images
         x(a, b) = clip(x₀ + a ε₁ v₁ + b ε₂ v₂, −1, 1),   a, b ∈ {−h, …, h},
     with ε_i = step_size · √λ_i by default ("perturbations scaled by the
     principal eigenvalues at the reference image", paper Sec. 6), or
     ε_i = step_size with --step-rule fixed.
  3. At every grid point compute M(x(a, b)) for BOTH the reference and the
     comparison encoder (ResNet-50 by default) and project it onto the fixed
     reference plane: M₂ = Vᵀ M V, V = [v₁, v₂]. Both encoders are therefore
     evaluated on the same images and read in the same 2-D basis.
  4. Convert each 2 × 2 matrix to ellipse parameters (width = 2√μ_max,
     height = 2√μ_min, angle of the major axis from the v₁ axis in degrees;
     see multigeom/geometry/metric_field.py for why this convention).

G is estimated with the conditional-pair estimator (multigeom.geometry.estimator) and
the settings of the packaged estimator config; J in closed form
(multigeom.geometry.fisher) with the same σ. The G eigenpairs at x₀ use the Stage-4 seed
(seed + img_idx), so they reproduce Stage 4's runs/per_image file for that image. Grid
point (i, j) uses its own generator (multigeom.geometry.metric_field.point_seed), shared
by the two encoders.

Outputs, one file per (metric, area):
    <out_dir>/{G,J}_{V1,V4}.pt
with matrices_2x2[encoder] [n, n, 2, 2] and ellipses[encoder] [n, n, 3]
indexed [i, j] ↔ (a, b) = (grid_coefficients[i], grid_coefficients[j]),
plus v1, v2, λ1, λ2, ε1, ε2, x0, the perturbed images [n, n, 64, 64],
settings and provenance. `figures/fig5.py` of multigeom-tutorials draws them.

Usage: python -m multigeom.pipeline.metric_field --img-idx 0 --out-dir
runs/metric_field/img0000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch


from ..config import config_path, load_config
from ..datasets import CANONICAL_TRANSFORM, TinyImageNetDataset
from ..diffusion.model import build_diffusion_model, build_scheduler
from ..encoders import build_encoder
from ..geometry.estimator import estimate_g
from ..geometry.fisher import fisher_eigendecomposition
from ..geometry.metric_field import (
    STEP_RULES,
    canonical_sign,
    ellipse_grid,
    grid_coefficients,
    perturbation_grid,
    point_seed,
    project_metric,
    step_sizes,
)
from ..utils import get_device, make_generator, provenance

ENCODERS = ("inception", "resnet50", "onoff")
ELLIPSE_CONVENTION = (
    "width = 2*sqrt(mu_max), height = 2*sqrt(mu_min) of the 2x2 "
    "matrix; angle (deg) of the major eigenvector from the v1 axis"
)


def parse_args(argv: list[str] | None = None):
    """Command-line arguments of `python -m multigeom.pipeline.metric_field`."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--img-idx",
        type=int,
        default=0,
        help="index into the TinyImageNet val list (same order as Stage 4)",
    )
    p.add_argument("--reference-encoder", default="inception", choices=ENCODERS)
    p.add_argument("--other-encoder", default="resnet50", choices=ENCODERS)
    p.add_argument("--areas", nargs="+", default=["V1", "V4"], choices=["V1", "V4"])
    p.add_argument("--metrics", nargs="+", default=["G", "J"], choices=["G", "J"])
    p.add_argument("--grid-size", type=int, default=5, help="odd n; the grid is n x n")
    p.add_argument(
        "--step-rule",
        default="sqrt_lambda",
        choices=STEP_RULES,
        help="sqrt_lambda: eps_i = step_size*sqrt(lambda_i); fixed: eps_i = step_size",
    )
    p.add_argument("--step-size", type=float, default=0.5)
    p.add_argument(
        "--runs-dir",
        default="runs",
        help="root with encoders/<enc>/<area>/best_model.pt and "
        "diffusion/<enc>/<area>/final_model.pt",
    )
    p.add_argument(
        "--config",
        default="estimator",
        help="estimator config: a packaged name or a YAML path",
    )
    p.add_argument("--diffusion-cfg", default="diffusion")
    p.add_argument(
        "--encoder-cfg-dir",
        default=None,
        help="directory with <encoder>.yaml (default: the packaged encoder configs)",
    )
    p.add_argument(
        "--out-dir", default=None, help="default: <runs-dir>/metric_field/img{idx:04d}"
    )
    p.add_argument("--device", default=None)
    p.add_argument("--data-dir", default=None, help="override NEURIPS_DATA_DIR")
    return p.parse_args(argv)


def load_models(
    encoder_name: str,
    area: str,
    runs_dir: Path,
    enc_cfg_dir: Path | None,
    diff_cfg,
    need_unet: bool,
    device: torch.device,
):
    """Encoder f_φ (and diffusion network ε_θ if needed) for one configuration."""
    enc_cfg_path = (
        config_path(encoder_name)
        if enc_cfg_dir is None
        else enc_cfg_dir / f"{encoder_name}.yaml"
    )
    enc_ckpt_path = runs_dir / "encoders" / encoder_name / area / "best_model.pt"
    diff_ckpt_path = runs_dir / "diffusion" / encoder_name / area / "final_model.pt"
    for path in [enc_cfg_path, enc_ckpt_path] + ([diff_ckpt_path] if need_unet else []):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; run Stages 1-3 for ({encoder_name}, {area}) first."
            )

    encoder = build_encoder(load_config(enc_cfg_path)).to(device)
    ckpt = torch.load(enc_ckpt_path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder.eval()

    unet = None
    if need_unet:
        unet = build_diffusion_model(diff_cfg).to(device)
        ckpt = torch.load(diff_ckpt_path, map_location=device, weights_only=False)
        unet.load_state_dict(ckpt["model_state_dict"])
        unet.eval()
    return encoder, unet, str(enc_cfg_path)


def eigendecomposition(
    metric: str,
    x: torch.Tensor,
    encoder,
    unet,
    alphas_cumprod,
    cfg,
    sigma: float,
    device: torch.device,
    seed: int | None,
):
    """Top-k eigenpairs (eigenvalues [k], eigenvectors [k, P]) of G(x) or J(x)."""
    if metric == "G":
        out = estimate_g(
            image=x,
            encoder=encoder,
            unet=unet,
            alphas_cumprod=alphas_cumprod,
            cfg=cfg,
            device=device,
            generator=make_generator(seed, device),
        )
    else:
        out = fisher_eigendecomposition(
            image=x,
            encoder=encoder,
            sigma=sigma,
            top_k=int(cfg.fisher_baseline.top_k),
            device=device,
        )
    return out.eigenvalues.cpu(), out.eigenvectors.cpu()


def main(argv: list[str] | None = None):
    """
    Compute the local metric field around a reference image (command-line entry point).
    """
    args = parse_args(argv)
    if args.reference_encoder == args.other_encoder:
        raise ValueError("--reference-encoder and --other-encoder must differ")
    if args.data_dir:
        os.environ["NEURIPS_DATA_DIR"] = args.data_dir

    cfg = load_config(args.config)
    diff_cfg = load_config(args.diffusion_cfg)
    device = get_device(args.device)
    runs_dir = Path(args.runs_dir)
    enc_cfg_dir = Path(args.encoder_cfg_dir) if args.encoder_cfg_dir else None
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else runs_dir / "metric_field" / f"img{args.img_idx:04d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    sigma = float(cfg.monte_carlo.observation_noise_sigma)
    sigma_train = float(diff_cfg.conditioning.observation_noise_sigma)
    if "G" in args.metrics and sigma != sigma_train:
        raise ValueError(
            f"Estimator σ={sigma} differs from the σ={sigma_train} the diffusion model "
            "was trained with; the observation model must be the same in both."
        )

    coeffs = grid_coefficients(args.grid_size)
    n = len(coeffs)
    seed = int(cfg.seed)
    encoders = (args.reference_encoder, args.other_encoder)

    dataset = TinyImageNetDataset(split="val", transform=CANONICAL_TRANSFORM)
    if not 0 <= args.img_idx < len(dataset):
        raise ValueError(f"--img-idx {args.img_idx} outside [0, {len(dataset)})")
    if args.img_idx >= int(cfg.n_test_images):
        print(
            f"[Stage 6] note: image {args.img_idx} is not among the first "
            f"{cfg.n_test_images} test images of Stage 4."
        )
    x0 = dataset[args.img_idx].unsqueeze(0).to(device)  # [1, 1, 64, 64] in [-1, 1]

    alphas_cumprod = build_scheduler(diff_cfg).alphas_cumprod.to(device)
    need_unet = "G" in args.metrics

    for area in args.areas:
        print(f"[Stage 6] {area}: loading {encoders[0]} and {encoders[1]} ...")
        models, cfg_paths = {}, {}
        for enc in encoders:
            encoder, unet, cfg_path = load_models(
                enc, area, runs_dir, enc_cfg_dir, diff_cfg, need_unet, device
            )
            models[enc] = (encoder, unet)
            cfg_paths[f"encoder_{enc}"] = cfg_path

        for metric in args.metrics:
            t0 = time.time()
            ref_encoder, ref_unet = models[encoders[0]]
            lam, vecs = eigendecomposition(
                metric,
                x0,
                ref_encoder,
                ref_unet,
                alphas_cumprod,
                cfg,
                sigma,
                device,
                seed + args.img_idx,
            )
            lam1, lam2 = float(lam[0]), float(lam[1])
            v1, v2 = canonical_sign(vecs[0]), canonical_sign(vecs[1])
            eps1, eps2 = step_sizes(lam1, lam2, args.step_rule, args.step_size)
            print(
                (
                    f"  {metric}: λ1={lam1:.4g}  λ2={lam2:.4g}  ε1={eps1:.4g}  "
                    f"ε2={eps2:.4g}"
                )
            )

            images, clamp_fraction = perturbation_grid(x0, v1, v2, eps1, eps2, coeffs)
            if clamp_fraction.max() > 0.05:
                print(
                    f"  warning: clipping to [-1, 1] changes up to "
                    f"{100 * float(clamp_fraction.max()):.0f}% of the pixels of a "
                    f"perturbed image; consider a smaller --step-size."
                )
            V = torch.stack([v1, v2]).to(device)  # [2, P]

            matrices, ellipses = {}, {}
            for enc in encoders:
                encoder, unet = models[enc]
                M = torch.empty(n, n, 2, 2, dtype=torch.float64)
                for i in range(n):
                    for j in range(n):
                        x = images[i, j].reshape(1, 1, *images.shape[-2:])
                        vals, evecs = eigendecomposition(
                            metric,
                            x,
                            encoder,
                            unet,
                            alphas_cumprod,
                            cfg,
                            sigma,
                            device,
                            point_seed(seed, args.img_idx, i, j, n),
                        )
                        M[i, j] = project_metric(vals, evecs.to(device), V).cpu()
                matrices[enc] = M
                ellipses[enc] = torch.from_numpy(ellipse_grid(M.numpy()))

            out_path = out_dir / f"{metric}_{area}.pt"
            torch.save(
                {
                    "img_idx": args.img_idx,
                    "metric": metric,
                    "area": area,
                    "reference_encoder": encoders[0],
                    "other_encoder": encoders[1],
                    "encoders": list(encoders),
                    "grid_coefficients": torch.from_numpy(coeffs),  # [n]
                    "eps1": eps1,
                    "eps2": eps2,
                    "v1": v1,  # [P]
                    "v2": v2,  # [P]
                    "lambda1": lam1,
                    "lambda2": lam2,
                    "reference_eigenvalues": lam,  # [k] at x0
                    "x0": x0.cpu(),  # [1, 1, 64, 64]
                    "perturbed_images": images.cpu(),  # [n, n, 64, 64]
                    "clamp_fraction": clamp_fraction,  # [n, n]
                    "matrices_2x2": matrices,  # {enc: [n, n, 2, 2]}
                    "ellipses": ellipses,  # {enc: [n, n, 3]}
                    "ellipse_convention": ELLIPSE_CONVENTION,
                    "settings": {
                        "grid_size": n,
                        "step_rule": args.step_rule,
                        "step_size": args.step_size,
                        "sigma": sigma,
                        "top_k_G": int(cfg.top_k),
                        "top_k_J": int(cfg.fisher_baseline.top_k),
                        "n_mc": int(cfg.monte_carlo.n_samples),
                        "reference_seed": seed + args.img_idx,
                        "grid_seed_rule": (
                            "multigeom.geometry.metric_field.point_seed"
                            "(seed, img_idx, i, j, n)"
                        ),
                    },
                    "elapsed_seconds": time.time() - t0,
                    "provenance": provenance(
                        {
                            "estimator": args.config,
                            "diffusion": args.diffusion_cfg,
                            **cfg_paths,
                        },
                        seed=seed,
                        argv=sys.argv
                        if argv is None
                        else ["06_metric_field.py", *argv],
                    ),
                },
                out_path,
            )
            print(f"  saved {out_path}  ({time.time() - t0:.0f} s)")

        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"[Stage 6] Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
