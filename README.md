# Multi-scale information geometry of neural population codes

Code for the NeurIPS 2026 paper

> **A multi-scale information geometry reveals the structure of mutual information in neural populations**
> Simone Azeglio, Steeve Laquitaine, Ulisse Ferrari, Matthew Chalk

The paper defines the *multi-scale Fisher information matrix*

$$G(x) = \int_0^\infty [\phi_t * J_t](x)\,dt ,$$

the Fisher information of the stimulus-diffused likelihood integrated over all
diffusion scales, and shows that it is directly related to the mutual
information between stimulus and response. This repository estimates the
leading eigenvectors of $G(x)$ for natural images from a conditional diffusion
model, compares them with those of the classical Fisher information
$J(x)$, and reproduces the macaque V1/V4 analyses of the paper (Figs. 3-5,
Table 1 and the Bures-Wasserstein appendix figure).

## What the pipeline does

1. **Encoders** (`stages/01_train_encoder.py`). A frozen backbone
   (InceptionV3 truncated at `Conv2d_4a_3x3`, ResNet-50 truncated at
   `layer1.0`, or the ON-OFF front-end of Berardino et al.) followed by a
   per-neuron Gaussian spatial-attention readout, fitted to the trial-averaged
   responses of the 50 most reliable V1 or V4 channels of the THINGS
   Ventral-stream Spiking Dataset (Papale et al., 2025).
2. **Synthetic responses** (`stages/02_precompute_responses.py`). The encoder
   $f_\phi$ is evaluated on every TinyImageNet training image; these 50-d
   vectors condition the diffusion model.
3. **Conditional diffusion model** (`stages/03_train_diffusion.py`). A
   `UNet2DConditionModel` (64×64 grayscale, ~68M parameters) conditioned on
   the responses through a 3-layer MLP and cross-attention. Training follows the
   Gaussian observation model $p(r\mid x)=\mathcal N(f_\phi(x),\sigma^2 I)$
   with $\sigma=0.22$, and uses classifier-free-guidance dropout (rate 0.1) so
   that the same network also gives the unconditional denoiser. 200 epochs.
4. **G(x) and J(x)** (`stages/04_estimate_gx.py`). The top-50 eigenpairs of
   $G(x)$ from the conditional-pair estimator (Algorithm 1 of the paper,
   `src/estimator.py`), and of $J(x)=\sigma^{-2}J_{f_\phi}^\top J_{f_\phi}$ by
   autograd (`src/fisher.py`), on 200 held-out TinyImageNet validation images.
5. **Cross-model comparison** (`stages/05_compute_metrics.py`). Subspace
   alignment $\mathcal A_k=\tfrac1k\|V_A^\top V_B\|_F^2$ and Bures-Wasserstein
   distance between every pair of (encoder, area) configurations, for
   $k=1,\dots,50$, averaged over images.
6. **Local metric field** (`stages/06_metric_field.py`). The 2×2 projection of
   $G$ and $J$ onto the leading two eigenvectors at a reference image, on a grid
   of images perturbed along those eigenvectors (Fig. 5).

Figures and the encoder table are drawn from these outputs by the scripts in
`figures/`. Supporting analyses quoted in the paper's appendices (calibration
of $\sigma$, Monte-Carlo variance of the conditional-pair versus the
unconditional-conditional estimator, backbone layer sweep) are in `analysis/`.

## Installation

```bash
git clone https://github.com/sazio/MultiscaleInformationGeometry.git
cd MultiscaleInformationGeometry
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt   # exact versions we ran (Python 3.11, CUDA 12.6)
pip install -e .                       # or: pip install -r requirements.txt
```

`requirements-lock.txt` pins the versions the pipeline and tests were run
with; `requirements.txt` / `pyproject.toml` declare the supported ranges.

## Data

See [`data/README.md`](data/README.md) for the three datasets (TVSD, THINGS
images, TinyImageNet) and the expected layout. Pretrained backbone weights are
downloaded automatically.

## Running

One (encoder, area) configuration end to end:

```bash
ENCODER=inception AREA=V1 bash reproduce.sh       # also: resnet50, onoff × V1, V4
```

All six configurations of the paper, then the comparisons and figures:

```bash
for area in V1 V4; do
  for enc in inception resnet50 onoff; do
    ENCODER=$enc AREA=$area bash reproduce.sh
  done
done
python stages/05_compute_metrics.py --runs-dir runs --out-dir runs/metrics
python stages/06_metric_field.py --img-idx 0 --out-dir runs/metric_field/img0000
python figures/table1.py     --runs-dir runs --out-dir figures/out
python figures/fig3.py       --runs-dir runs --out-dir figures/out --img-idx 0
python figures/fig4.py       --runs-dir runs --out-dir figures/out --img-idx 0
python figures/fig5.py       --runs-dir runs --out-dir figures/out --img-idx 0
python figures/fig_app_bw.py --runs-dir runs --out-dir figures/out
```

Stage 4 accepts `--start/--end` to split the 200 test images across GPUs.
Each image is seeded independently (`seed + image index`), so the split does
not change the results. Every checkpoint and output file stores a provenance
record (git commit, command line, seed, config files, package versions).

## Where each result comes from

| Paper item | Script | Inputs |
|---|---|---|
| Table 1 (encoder performance) | `figures/table1.py` | `runs/encoders/*/*/best_model.pt` |
| Fig. 3B-C (measured vs predicted responses) | `figures/fig3.py` | stage 1 |
| Fig. 3D (receptive fields) | `figures/fig3.py` | stage 1 readout parameters |
| Fig. 3E-F (eigenvectors of G and J) | `figures/fig3.py` | stage 4 |
| Fig. 4A (performance of the three encoders) | `figures/fig4.py` | stage 1 |
| Fig. 4B-C (eigenvectors per encoder) | `figures/fig4.py` | stage 4 |
| Fig. 4D-E (subspace alignment vs k) | `figures/fig4.py` | stage 5 |
| Fig. 5 (local metric field) | `figures/fig5.py` | stage 6 |
| Bures-Wasserstein figure (appendix) | `figures/fig_app_bw.py` | stage 5 |
| σ = 0.22 calibration (appendix) | `analysis/calibrate_sigma.py` | TVSD |
| CC vs UC estimator variance (appendix) | `analysis/estimator_variance.py` | stages 1, 3 |
| Backbone layer choice (appendix) | `analysis/layer_sweep.py` | TVSD, THINGS |

Fig. 3A (model schematic) and the illustrative one- and two-dimensional
examples of Figs. 1 and 2 are not produced by this code.

## Compute

Measured on NVIDIA A100-SXM4-40GB GPUs, per (encoder, area) configuration:

| Stage | Time |
|---|---|
| 1. encoder readout (features cached once) | minutes |
| 2. TinyImageNet responses | minutes |
| 3. diffusion training, 200 epochs | ~30 A100-hours (~15 h on 2 GPUs) |
| 4. G(x) and J(x), 200 images | ~2 A100-hours |
| 5. metrics | minutes (CPU) |

## Tests

```bash
pytest                       # fast suite, CPU, no downloads
pytest -m "slow or pretrained"
```

## Checkpoints

Trained encoder and diffusion checkpoints, and the per-image G(x)/J(x)
eigendecompositions behind the figures, will be released with a persistent
identifier; this section will link to them.

## Repository layout

```
configs/      encoder, diffusion and estimator settings (one YAML per component)
src/          library: data loading, encoders, diffusion model, estimator, Fisher, metrics
stages/       numbered pipeline scripts 01-06
figures/      figure and table scripts
analysis/     supporting analyses quoted in the appendices
tests/        pytest suite
data/         dataset location (see data/README.md)
reproduce.sh  stages 1-4 for one (encoder, area) configuration
```

## Citation

See [`CITATION.cff`](CITATION.cff).

```bibtex
@inproceedings{azeglio2026multiscale,
  title     = {A multi-scale information geometry reveals the structure of mutual information in neural populations},
  author    = {Azeglio, Simone and Laquitaine, Steeve and Ferrari, Ulisse and Chalk, Matthew},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

Code: MIT (see `LICENSE`). The datasets keep their own licenses: TVSD
(Papale et al.), THINGS images (Hebart et al.), TinyImageNet. The ON-OFF
front-end and its pretrained parameters come from `plenoptic`; the
conditional UNet from Hugging Face `diffusers`.
