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

## Bring your own encoder

Any fitted PyTorch model can be used as the encoder f(x), if it follows these rules. `src/encoder_contract.py`
checks them (`validate_encoder(encoder)`; `fisher_eigendecomposition` does it automatically for models that
do not use the reference-encoder internals, and `conditional_pair_estimate(..., validate=True)` on request) and
raises `EncoderContractError` naming the rule.

| Rule | Requirement |
|---|---|
| R1 | an `nn.Module` |
| R2 | input `[B, 1, 64, 64]`, values in [-1, 1]; the model does any rescaling or channel repeat itself |
| R3 | output `[B, N]`, floating point and finite, the same N for every batch size (and `n_neurons == N` if set) |
| R4 | deterministic in eval mode (no dropout or noise at inference) |
| R5 | differentiable w.r.t. the input through `forward` (no `torch.no_grad()` or `detach()`): needed for J(x) |
| R6 | samples independent within a batch (no batch statistics in train mode) |
| R7 | does not modify its input in place |
| R8 | responses vary across images (warning) |
| R9 | responses of order 1 (for example z-scored), as the noise sigma is 0.22 (warning) |

Training the diffusion model on the encoder's responses needs R1-R4, R6 and R9; estimating G(x) and J(x)
needs all of them. The fit itself, and the 64x64 grayscale natural-image domain, are not checked.
The diffusion model's conditioning layer is sized by N, so each encoder needs its own trained diffusion model.

## Subspace alignment (Fig. 4D-E) on several GPUs

`tutorials/2_subspace_alignment.ipynb` estimates G(x) and J(x) for many images and configurations and
plots Fig. 4D-E. The images are independent, so they are split over workers (local GPUs, or serverless
GPUs on [Modal](https://modal.com)) with identical results; no DDP/FSDP is involved. All supporting
functions are in `src/subspace_alignment.py`. The notebook sets five parameters, `CONFIGS` (one or more
(encoder, area) pairs), `N_IMAGES`, `BACKEND` (`"local"` or `"modal"`), `DEVICE` (`"gpu"` or `"cpu"`) and
`NUM_WORKERS`, and a table in the notebook gives example values for four typical cases. The Modal route
needs `pip install modal` and `modal token new`, and costs money: the notebook prints an estimate and
refuses to launch above `MAX_COST_USD`.

## Local metric field (Fig. 5)

`tutorials/3_local_metric_field.ipynb` plots the local metric field projected onto the leading eigenvectors of
G(x): a reference image is perturbed along the two leading eigenvectors of the Inception model's G(x), and the
metrics G(x) and J(x) of the Inception and ResNet-50 models are drawn as ellipses on that 2-D grid, for V1 and V4.
One GPU is enough: every grid point of every model needs one G(x) estimate (about 15 s on an RTX 5090), so the 5 x 5
grid with both areas takes about 26 minutes and a 3 x 3 grid with one area about 5 minutes. The computation is
`stages/06_metric_field.py`, run as a background process by `src/local_metric_field.py`, and it can be stopped
and resumed. The notebook perturbs by `0.5*sqrt(lambda)` for G(x) (about 0.3) but by a fixed 0.3 for J(x): J has far
larger eigenvalues, and with the same rule its perturbed images are clipped and the ellipses of the two models
swap sizes, unlike the paper's panel B. The paper does not state its step sizes, so this is a choice (the stage-6
script itself uses `0.5*sqrt(lambda)` for both metrics, and the notebook runs it once per metric). The paper does not state
its reference image either; the notebook uses validation image 64, the one that best reproduces the pattern of the
paper's panel B among the 200 test images (`tools/scan_metric_field_images.py`, result in
`analysis/results/metric_field_scan.jsonl`), whereas the stage-6 script defaults to image 0. The elongated V1 ellipses
of the paper's panel A are not reproduced (the leading eigenvalues of G(x) are close for every image).

## Tests

```bash
pytest tests/unit                                        # unit tests: synthetic data, no checkpoints
pytest tests/integration                                 # notebook reproduction of Fig. 3D-F and Fig. 4B-C
pytest -m "not slow and not gpu and not pretrained"      # what GitHub runs on every pull request
```

The integration tests compare the tutorial notebook's plots with references stored in
`tests/integration/fixtures/`. The plot comparison runs anywhere; the "recompute" tests (marked
`slow`) also need the checkpoints in `runs/` and the TinyImageNet data, and skip themselves
without them.

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
tests/unit/        unit tests (synthetic data)
tests/integration/ notebook reproduction of Fig. 3D-F, Fig. 4B-C and Fig. 4D-E against stored references
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
