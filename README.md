# Multi-scale information geometry of neural population codes

Code accompanying the NeurIPS 2026 submission *"Multi-scale information geometry uncovers the structure of mutual information in neural population codes"* (anonymous).

This repository implements the full pipeline used in the paper: training image-to-response encoders on macaque V1 / V4 recordings, training conditional diffusion models on natural images, estimating the **multi-scale Fisher information matrix** $G(x)$ via the **conditional-pair estimator** (Algorithm 1 of the paper), and computing the cross-encoder subspace-alignment and Bures-Wasserstein metrics that make up Fig. 4.

## Pipeline

The pipeline is organised in five numbered stages, run as standalone scripts in `stages/`:

| # | Script | Purpose |
|---|--------|---------|
| 1 | `stages/01_train_encoder.py` | Fit per-neuron Gaussian spatial-attention readout on a frozen backbone (Inception-V3, ResNet-50, or ON-OFF), targeting the 50 most reliable channels of macaque V1 or V4 from the THINGS Ventral-stream Spiking Dataset. |
| 2 | `stages/02_precompute_responses.py` | Generate synthetic neural responses for every TinyImageNet image by evaluating the trained encoder; these form the conditioning vectors for the diffusion model. |
| 3 | `stages/03_train_diffusion.py` | Train a conditional UNet (`UNet2DConditionModel` from `diffusers`) on TinyImageNet conditioned on the responses from Stage 2, with classifier-free-guidance dropout. |
| 4 | `stages/04_estimate_gx.py` | Compute the top-$k$ eigenpairs of $G(x)$ on a held-out test set of $200$ images using the **conditional-pair estimator** of Algorithm 1, and the closed-form Fisher baseline $J(x) = \sigma^{-2} J_{f_\phi}^\top J_{f_\phi}$ via PyTorch autograd. |
| 5 | `stages/05_compute_metrics.py` | Aggregate cross-encoder subspace alignment $\mathcal{A}_k = \frac{1}{k}\|V_A^\top V_B\|_F^2$ and Bures-Wasserstein distance $d_{\mathrm{BW}}$ across (encoder, area) pairs (Fig. 4D-E and Fig. App-BW). |

The whole pipeline for one (encoder, area) configuration runs end-to-end in `reproduce.sh`.

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download data (one-time; see data/README.md for sources & instructions)
#    - Papale TVSD: monkeyN_THINGS_normMUA.mat + things_imgs.mat
#    - THINGS images
#    - TinyImageNet
# Place under data/ as described.

# 3. Run the canonical (Inception, V1) configuration end-to-end:
bash reproduce.sh
```

The full pipeline takes roughly **30 A100-GPU-hours** for one (encoder, area) configuration: ~0.5h encoder readout + ~0.1h response extraction + ~15h diffusion training + ~0.3h G(x) estimation. The metrics script is sub-minute.

To reproduce Fig. 4D-E in full you need all 6 configurations: `{Inception, ResNet-50, ON-OFF} × {V1, V4}`. Each can be launched independently, in parallel.

## Layout

```
neurips_code/
├── README.md                     <- this file
├── requirements.txt
├── LICENSE                       <- MIT
├── reproduce.sh                  <- end-to-end driver for one (encoder, area)
├── configs/
│   ├── inception.yaml            <- Inception-V3 / Conv2d_4a_3x3 backbone
│   ├── resnet50.yaml             <- ResNet-50 / layer1.0 backbone
│   ├── onoff.yaml                <- frozen Berardino ON-OFF front-end
│   ├── diffusion.yaml            <- DDPM hyperparameters (shared)
│   └── estimator.yaml            <- conditional-pair estimator config (shared)
├── data/
│   └── README.md                 <- where to obtain each dataset
├── src/                          <- importable library
│   ├── data_loaders.py           <- TVSD + TinyImageNet loaders
│   ├── encoders.py               <- backbone + Gaussian-readout encoder classes
│   ├── diffusion_model.py        <- conditional UNet + scheduler wrapper
│   ├── estimator.py              <- conditional-pair G(x) estimator (Alg. 1)
│   ├── fisher.py                 <- closed-form Fisher J(x) baseline
│   ├── metrics.py                <- subspace alignment + Bures-Wasserstein
│   └── utils.py                  <- config loader, seeding, device
└── stages/
    ├── 01_train_encoder.py
    ├── 02_precompute_responses.py
    ├── 03_train_diffusion.py
    ├── 04_estimate_gx.py
    └── 05_compute_metrics.py
```

## Reproducing Figure 4D-E

Fig. 4D-E shows cross-encoder agreement of $G(x)$ vs $J(x)$ for V1 and V4. To reproduce:

```bash
# (a) Train all 6 encoders
for area in V1 V4; do
  for cfg in inception resnet50 onoff; do
    python stages/01_train_encoder.py \
      --config configs/$cfg.yaml --area $area \
      --out-dir runs/encoders/$cfg/$area
  done
done

# (b) Precompute TinyImageNet responses for all 6
for area in V1 V4; do
  for cfg in inception resnet50 onoff; do
    python stages/02_precompute_responses.py \
      --config configs/$cfg.yaml --area $area \
      --encoder-ckpt runs/encoders/$cfg/$area/best_model.pt \
      --out-dir runs/responses/$cfg/$area
  done
done

# (c) Train all 6 diffusion models (~15h each on 1 A100)
for area in V1 V4; do
  for cfg in inception resnet50 onoff; do
    python stages/03_train_diffusion.py \
      --config configs/diffusion.yaml \
      --responses runs/responses/$cfg/$area/train_responses.pt \
      --out-dir runs/diffusion/$cfg/$area
  done
done

# (d) Estimate G(x) and J(x) on the 200 test images
for area in V1 V4; do
  for cfg in inception resnet50 onoff; do
    python stages/04_estimate_gx.py \
      --config configs/estimator.yaml \
      --encoder-cfg configs/$cfg.yaml \
      --encoder-ckpt runs/encoders/$cfg/$area/best_model.pt \
      --diffusion-ckpt runs/diffusion/$cfg/$area/best_model.pt \
      --out-dir runs/per_image/$cfg/$area
  done
done

# (e) Aggregate metrics for the figure
python stages/05_compute_metrics.py --runs-dir runs --out-dir runs/metrics
```

## Compute requirements

All training and inference fits comfortably on one **NVIDIA A100 (40GB)**. Multi-GPU is not required.

| Stage | Per-config wall time on 1×A100 |
|-------|---------------------------------|
| 01 (encoder readout) | ~30 min |
| 02 (response precompute) | ~5 min |
| 03 (diffusion training, 200 epochs) | ~15 hours |
| 04 ($G(x)$ + $J(x)$ on 200 images) | ~20 min |
| 05 (metrics aggregation) | <1 min |
| **Total per (encoder, area)** | **~16 hours** |
| **Full Fig. 4 (6 configs)** | **~96 GPU-hours** |

## License

MIT — see `LICENSE`.

## Datasets and dependencies

External assets used under their published licenses:

- **THINGS Ventral-stream Spiking Dataset** (Papale et al.) — macaque V1/V4 normMUA.
- **THINGS image database** (Hebart et al.) — natural-image stimuli.
- **TinyImageNet** — diffusion training images.
- **`plenoptic`** library — provides the Berardino-2017 ON-OFF frontend.
- **HuggingFace `diffusers`** — provides `UNet2DConditionModel`.

See `data/README.md` for download instructions.
