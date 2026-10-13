"""
Monte-Carlo variance of the conditional-pair (CC) estimator of G(x) versus the
unconditional-conditional (UC) form (paper App. "Conditional-pair estimator"):

    "On a small held-out set of test images with multiple independent Monte
     Carlo realisations per image, the CC estimator yields top-5 eigenvalues of
     G(x) with ~80-95% lower Monte Carlo variance than the UC form at matched
     compute; the UC top eigenvalue further differs from the CC value by a
     per-image-dependent factor of roughly 5-50x."

The two estimators
------------------
Both follow Algorithm 1 (multigeom.geometry.estimator.conditional_pair_estimate): the same
log-τ grid and trapezoid weights w_j, the same forward diffusion
x_t = √ᾱ_t x + √(1−ᾱ_t) z, the same null-token Tweedie denoising
x̂ = clip((x_t − √(1−ᾱ_t) ε_θ(x_t,t,0)) / √ᾱ_t, −1, 1), and the same response
sampler r = f_φ(x̂) + σ ξ. They differ only in the row appended to D:

    CC:  √(w_j / (2 N_MC)) · (ε_θ(x_t,t,r_1) − ε_θ(x_t,t,r_2)),   r_1, r_2 i.i.d.
    UC:  √(w_j /    N_MC ) · (ε_θ(x_t,t,r)   − ε_θ(x_t,t,0)),     one r

Scaling of the UC rows. Eq. (eq:line) writes the diffused Fisher as
(1/τ²)·Cov_r[x̂(x_t,r)], and x̂(x_t,r) − x̂(x_t,r') = −√τ (ε(r) − ε(r')) in the
ε-parametrisation. For i.i.d. r_1, r_2, E[(ε_1−ε_2)(ε_1−ε_2)ᵀ] = 2 Cov_r[ε(r)],
hence the 1/2 in CC. The UC form instead centres one draw on the
unconditional prediction. If the models are correct and r is drawn from the
diffused likelihood, the tower property gives E_r[ε(x_t,t,r)] = ε(x_t,t,0), so

    E[(ε(r) − ε(0))(ε(r) − ε(0))ᵀ] = Cov_r[ε(r)],

with no factor 2: the UC row weight is √(w_j/N_MC), and E[D_UCᵀ D_UC] =
E[D_CCᵀ D_CC] = Σ_j w_j E_z Cov_r[ε]. When the null-token prediction is not the
mean of the conditional predictions (imperfect CFG training, or the
point-mass sampler r ~ p(r | x̂(x_t)) instead of p_t(r | x_t)), the UC form
picks up an additional term Σ_j w_j E_z[b_j b_jᵀ] with
b_j = E_r[ε(x_t,t,r)] − ε(x_t,t,0); the CC form is free of it because b_j is
common to both draws and cancels in the subtraction.

Compute. With the repo's Tweedie sampler the null-token pass is needed for x̂
anyway; UC reuses it as its subtraction term, so one UC Monte-Carlo sample
costs 2 UNet passes (null + one conditional) and one CC sample costs 3 (null +
two conditionals). `--compute-match passes` (default) gives UC
round(1.5·N_MC / B)·B samples per timestep so both estimators spend the same
number of UNet passes; `--compute-match samples` uses the same N_MC for both
(the convention of the original measurement script, whose sampler took
responses from the clean image and therefore spent 2 passes on each
estimator).

Output (JSON): for each image the top-k eigenvalues of every realisation of
both estimators, their across-run mean, variance and CV² = Var/mean², the
per-eigenvalue reductions (Var_UC − Var_CC)/Var_UC and (CV²_UC − CV²_CC)/CV²_UC,
the ratio mean λ_1^UC / mean λ_1^CC, and the top-k subspace alignment between
the CC and UC eigenvectors (averaged over runs); plus aggregate medians and a
provenance record. The raw-variance reduction mixes noise with the overall
scale of the two estimates (UC eigenvalues are larger); CV² is scale-free.

Usage:
    python -m multigeom.pipeline.estimator_variance \\
        --encoder-cfg inception \\
        --encoder-ckpt runs/encoders/inception/V1/best_model.pt \\
        --diffusion-ckpt runs/diffusion/inception/V1/final_model.pt \\
        --n-images 6 --n-runs 6 --n-mc 100 \\
        --out runs/analysis/estimator_variance_inception_V1.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


from ..geometry.estimator import (  # noqa: E402
    EstimatorOutput, _build_log_tau_grid, _trapezoid_weights, conditional_pair_estimate,
)
from ..config import load_config  # noqa: E402
from ..utils import get_device, make_generator, provenance  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


# ────────────────────────────────────────────────────────────────────────────
# UC estimator
# ────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def unconditional_conditional_estimate(
    image: torch.Tensor,
    encoder,
    unet,
    alphas_cumprod: torch.Tensor,
    cfg,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> EstimatorOutput:
    """
    Top-k eigenpairs of G(x) from the UC form: Algorithm 1 with the row
    √(w_j/N_MC)·(ε_θ(x_t,t,r) − ε_θ(x_t,t,0)) in place of the conditional pair.

    Same inputs and conventions as `multigeom.geometry.estimator.conditional_pair_estimate`
    (`cfg.monte_carlo.n_samples` is this estimator's N_MC).
    """
    encoder.eval()
    unet.eval()
    if image.dim() != 4 or image.shape[0] != 1:
        raise ValueError(f"Expected single-image batch [1, C, H, W], got {tuple(image.shape)}")
    image = image.to(device)
    H, W = image.shape[-2:]
    P = H * W

    grid_t = list(range(
        cfg.monte_carlo.timestep_grid.start,
        cfg.monte_carlo.timestep_grid.stop + 1,
        cfg.monte_carlo.timestep_grid.step,
    ))
    w = _trapezoid_weights(_build_log_tau_grid(grid_t, alphas_cumprod))
    N_MC = int(cfg.monte_carlo.n_samples)
    B = int(cfg.monte_carlo.batch_size)
    sigma = float(cfg.monte_carlo.observation_noise_sigma)
    if N_MC % B != 0:
        raise ValueError(f"N_MC={N_MC} must be divisible by batch_size={B}")

    D = torch.empty(len(grid_t) * N_MC, P, dtype=torch.float32)
    row = 0
    null_token = torch.zeros(B, encoder.n_neurons, device=device)
    for j, t_j in enumerate(grid_t):
        s_j = float((w[j] / N_MC).sqrt())                     # no 1/2: single centred draw
        sqrt_ab = float(alphas_cumprod[t_j].sqrt())
        sqrt_1m = float((1.0 - alphas_cumprod[t_j]).sqrt())
        t_tensor = torch.full((B,), t_j, device=device, dtype=torch.long)
        for _ in range(N_MC // B):
            z = torch.randn(B, 1, H, W, device=device, generator=generator)
            x_t = sqrt_ab * image + sqrt_1m * z
            eps_u = unet(x_t, t_tensor, null_token)             # reused below
            x_hat = ((x_t - sqrt_1m * eps_u) / sqrt_ab).clamp(-1.0, 1.0)
            mu = encoder(x_hat)
            xi = torch.randn(mu.shape, device=device, dtype=mu.dtype, generator=generator)
            eps_c = unet(x_t, t_tensor, mu + sigma * xi)
            D[row:row + B] = (s_j * (eps_c - eps_u).reshape(B, P)).cpu()
            row += B

    _, S, Vh = torch.linalg.svd(D, full_matrices=False)
    k = int(cfg.top_k)
    return EstimatorOutput(eigenvalues=(S ** 2)[:k].clone(), eigenvectors=Vh[:k].clone())


# ────────────────────────────────────────────────────────────────────────────
# Study
# ────────────────────────────────────────────────────────────────────────────
def uc_samples_for_matched_passes(n_mc_cc: int, batch_size: int) -> int:
    """UC sample count with the same number of UNet passes as CC (3 vs 2 per sample)."""
    return max(batch_size, int(round(1.5 * n_mc_cc / batch_size)) * batch_size)


def _with_n_mc(cfg, n_mc: int, top_k: int):
    c = copy.deepcopy(cfg)
    c.monte_carlo.n_samples = int(n_mc)
    c.top_k = int(top_k)
    return c


def _summary(runs: np.ndarray) -> dict:
    """runs: [n_runs, k] → across-run mean, variance (ddof=1) and CV²."""
    mean = runs.mean(axis=0)
    var = runs.var(axis=0, ddof=1)
    return {"mean": mean, "var": var, "cv2": var / np.maximum(mean ** 2, 1e-30)}


def run_variance_study(
    images: torch.Tensor,
    encoder,
    unet,
    alphas_cumprod: torch.Tensor,
    cfg,
    device: torch.device,
    n_runs: int,
    top_k: int = 5,
    n_mc_uc: int | None = None,
    seed: int = 12345,
    log=print,
) -> dict:
    """
    Run both estimators `n_runs` times on every image with independent seeds.

    Args:
        images: [n_images, 1, H, W] in [-1, 1].
        cfg: estimator config; cfg.monte_carlo.n_samples is the CC N_MC.
        n_mc_uc: UC samples per timestep (default: same as CC).
        seed: run r on image i uses generator seed `seed + 1000·i + r` for both
              estimators (the streams are consumed differently by the two).
    Returns:
        dict with per-image results and aggregate statistics (JSON-serialisable).
    """
    if n_runs < 2:
        raise ValueError("need at least 2 runs per image to estimate a variance")
    n_mc_cc = int(cfg.monte_carlo.n_samples)
    n_mc_uc = n_mc_cc if n_mc_uc is None else int(n_mc_uc)
    cfg_cc = _with_n_mc(cfg, n_mc_cc, top_k)
    cfg_uc = _with_n_mc(cfg, n_mc_uc, top_k)

    per_image = []
    for i in range(images.shape[0]):
        x = images[i:i + 1]
        ev = {"CC": [], "UC": []}
        align = []
        for r in range(n_runs):
            s = seed + 1000 * i + r
            cc = conditional_pair_estimate(x, encoder, unet, alphas_cumprod, cfg_cc, device,
                                           generator=make_generator(s, device))
            uc = unconditional_conditional_estimate(x, encoder, unet, alphas_cumprod, cfg_uc,
                                                    device, generator=make_generator(s, device))
            ev["CC"].append(cc.eigenvalues.double().numpy())
            ev["UC"].append(uc.eigenvalues.double().numpy())
            C = cc.eigenvectors.double() @ uc.eigenvectors.double().T
            align.append(float((C ** 2).sum()) / C.shape[0])
            log(f"  image {i} run {r}: CC λ={np.array2string(ev['CC'][-1], precision=3)}  "
                f"UC λ={np.array2string(ev['UC'][-1], precision=3)}")
        st = {m: _summary(np.stack(ev[m])) for m in ("CC", "UC")}
        var_red = (st["UC"]["var"] - st["CC"]["var"]) / np.maximum(st["UC"]["var"], 1e-30)
        cv2_red = (st["UC"]["cv2"] - st["CC"]["cv2"]) / np.maximum(st["UC"]["cv2"], 1e-30)
        per_image.append({
            "eigenvalues_CC": np.stack(ev["CC"]).tolist(),
            "eigenvalues_UC": np.stack(ev["UC"]).tolist(),
            "mean_CC": st["CC"]["mean"].tolist(), "mean_UC": st["UC"]["mean"].tolist(),
            "var_CC": st["CC"]["var"].tolist(), "var_UC": st["UC"]["var"].tolist(),
            "cv2_CC": st["CC"]["cv2"].tolist(), "cv2_UC": st["UC"]["cv2"].tolist(),
            "variance_reduction": var_red.tolist(),
            "cv2_reduction": cv2_red.tolist(),
            "ratio_top_eigenvalue_UC_over_CC": float(st["UC"]["mean"][0] / st["CC"]["mean"][0]),
            "alignment_CC_UC_topk": float(np.mean(align)),
        })

    def agg(key):
        """The per-image values of `key`, as an array."""
        return np.array([p[key] for p in per_image])

    var_red, cv2_red = agg("variance_reduction"), agg("cv2_reduction")
    ratios = agg("ratio_top_eigenvalue_UC_over_CC")
    return {
        "n_images": int(images.shape[0]),
        "n_runs": int(n_runs),
        "top_k": int(top_k),
        "n_mc_cc": n_mc_cc,
        "n_mc_uc": n_mc_uc,
        "unet_passes_per_timestep": {"CC": 3 * n_mc_cc, "UC": 2 * n_mc_uc},
        "per_image": per_image,
        "aggregate": {
            "median_variance_reduction_percent": float(100 * np.median(var_red)),
            "median_variance_reduction_percent_per_k": (100 * np.median(var_red, axis=0)).tolist(),
            "median_cv2_reduction_percent": float(100 * np.median(cv2_red)),
            "median_cv2_reduction_percent_per_k": (100 * np.median(cv2_red, axis=0)).tolist(),
            "ratio_top_eigenvalue_UC_over_CC": {
                "median": float(np.median(ratios)), "min": float(ratios.min()),
                "max": float(ratios.max()),
            },
        },
    }


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None):
    """Command-line arguments of `python -m multigeom.pipeline.estimator_variance`."""
    p = argparse.ArgumentParser(description="CC vs UC Monte-Carlo variance of G(x)")
    p.add_argument("--config", default=str(REPO / "configs" / "estimator.yaml"))
    p.add_argument("--encoder-cfg", required=True)
    p.add_argument("--encoder-ckpt", default=None,
                   help="Stage-1 checkpoint (omit only with --random-init)")
    p.add_argument("--diffusion-cfg", default=str(REPO / "configs" / "diffusion.yaml"))
    p.add_argument("--diffusion-ckpt", default=None,
                   help="Stage-3 checkpoint (omit only with --random-init)")
    p.add_argument("--random-init", action="store_true",
                   help="smoke test: skip loading checkpoints (random readout and UNet)")
    p.add_argument("--image-dir", default=None,
                   help="folder of images (sorted, first --n-images used); "
                        "default: TinyImageNet val split as in Stage 4")
    p.add_argument("--n-images", type=int, default=6)
    p.add_argument("--n-runs", type=int, default=6)
    p.add_argument("--n-mc", type=int, default=100, help="CC Monte-Carlo samples per timestep")
    p.add_argument("--batch-size", type=int, default=None, help="override cfg batch size")
    p.add_argument("--timestep-grid", type=int, nargs=3, default=None,
                   metavar=("START", "STOP", "STEP"), help="override cfg timestep grid")
    p.add_argument("--compute-match", choices=["passes", "samples"], default="passes")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out", default="runs/analysis/estimator_variance.json")
    return p.parse_args(argv)


def _load_images(args) -> torch.Tensor:
    from PIL import Image
    from ..data import CANONICAL_TRANSFORM, TinyImageNetDataset

    if args.image_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        files = sorted(p for p in Path(args.image_dir).rglob("*") if p.suffix.lower() in exts)
        if len(files) < args.n_images:
            raise ValueError(f"{args.image_dir} has {len(files)} images, need {args.n_images}")
        return torch.stack([CANONICAL_TRANSFORM(Image.open(f).convert("RGB"))
                            for f in files[:args.n_images]])
    ds = TinyImageNetDataset(split="val", transform=CANONICAL_TRANSFORM)
    return torch.stack([ds[i] for i in range(args.n_images)])


def main(argv: list[str] | None = None) -> dict:
    """Compare the Monte-Carlo variance of the two estimators (command-line entry point)."""
    from ..diffusion.model import build_diffusion_model, build_scheduler
    from ..encoders import build_encoder

    args = parse_args(argv)
    if args.data_dir:
        os.environ["NEURIPS_DATA_DIR"] = args.data_dir
    if not args.random_init and (args.encoder_ckpt is None or args.diffusion_ckpt is None):
        raise SystemExit("--encoder-ckpt and --diffusion-ckpt are required without --random-init")
    device = get_device(args.device)
    torch.manual_seed(args.seed)                     # random-init weights only

    cfg = load_config(args.config)
    cfg.monte_carlo.n_samples = args.n_mc
    if args.batch_size is not None:
        cfg.monte_carlo.batch_size = args.batch_size
    if args.timestep_grid is not None:
        g = cfg.monte_carlo.timestep_grid
        g.start, g.stop, g.step = args.timestep_grid
    diff_cfg = load_config(args.diffusion_cfg)
    enc_cfg = load_config(args.encoder_cfg)
    if float(cfg.monte_carlo.observation_noise_sigma) != float(
            diff_cfg.conditioning.observation_noise_sigma):
        raise ValueError("estimator and diffusion σ differ; the observation model must match")

    encoder = build_encoder(enc_cfg).to(device)
    unet = build_diffusion_model(diff_cfg).to(device)
    if not args.random_init:
        encoder.load_state_dict(torch.load(args.encoder_ckpt, map_location=device,
                                           weights_only=False)["model_state_dict"])
        unet.load_state_dict(torch.load(args.diffusion_ckpt, map_location=device,
                                        weights_only=False)["model_state_dict"])
    alphas_cumprod = build_scheduler(diff_cfg).alphas_cumprod.to(device)

    images = _load_images(args)
    B = int(cfg.monte_carlo.batch_size)
    n_mc_uc = (uc_samples_for_matched_passes(args.n_mc, B)
               if args.compute_match == "passes" else args.n_mc)
    print(f"[estimator_variance] {args.n_images} images x {args.n_runs} runs; "
          f"N_MC: CC={args.n_mc}, UC={n_mc_uc} ({args.compute_match}-matched)")

    t0 = time.time()
    result = run_variance_study(images, encoder, unet, alphas_cumprod, cfg, device,
                                n_runs=args.n_runs, top_k=args.top_k, n_mc_uc=n_mc_uc,
                                seed=args.seed)
    result["compute_match"] = args.compute_match
    result["random_init"] = bool(args.random_init)
    result["elapsed_seconds"] = time.time() - t0
    result["provenance"] = provenance(
        {"estimator": args.config, "encoder": args.encoder_cfg, "diffusion": args.diffusion_cfg},
        seed=args.seed, argv=argv)

    a = result["aggregate"]
    print(f"[estimator_variance] median variance reduction (CC vs UC, top-{args.top_k}): "
          f"{a['median_variance_reduction_percent']:.1f}%   "
          f"CV² reduction: {a['median_cv2_reduction_percent']:.1f}%   "
          f"λ1 UC/CC: median {a['ratio_top_eigenvalue_UC_over_CC']['median']:.2f} "
          f"[{a['ratio_top_eigenvalue_UC_over_CC']['min']:.2f}, "
          f"{a['ratio_top_eigenvalue_UC_over_CC']['max']:.2f}]")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[estimator_variance] saved {out}")
    return result


if __name__ == "__main__":
    main()
