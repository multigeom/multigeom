<p align="center">
  <img src="https://raw.githubusercontent.com/multigeom/multigeom/main/docs/source/_static/img/logo.png" alt="multigeom" width="320">
</p>

# multigeom

Multi-scale information geometry of neural population codes, in Python.

`multigeom` computes the Fisher information metric J(x) of any fitted PyTorch model of neural responses, and the
*multi-scale Fisher metric*

    G(x) = ∫₀^∞ [φ_t ∗ J_t](x) dt

the Fisher information of the stimulus-diffused likelihood integrated over all diffusion scales. G(x) is directly
related to the mutual information between stimulus and response, and for natural images it is estimated with a
conditional diffusion model. The package also reproduces the macaque V1/V4 analyses of the paper.

- **Landing page:** <https://multigeom.github.io>
- **Documentation and demos:** <https://multigeom.github.io/multigeom/>
- **Paper:** *A multi-scale information geometry reveals the structure of mutual information in neural populations*,
  Azeglio, Laquitaine, Ferrari, Chalk, NeurIPS 2026 ([arXiv:2605.06304](https://arxiv.org/abs/2605.06304))
- **Tutorials:** [`multigeom-tutorials`](https://github.com/multigeom/multigeom-tutorials), notebooks from "bring your own
  encoder" to the figures of the paper
- **Paper code (frozen record of the published results):**
  [`MultiscaleInformationGeometry`](https://github.com/sazio/MultiscaleInformationGeometry)

## What the pipeline does

1. **Encoders** (`multigeom.fit.train_encoder`). A frozen backbone
   (InceptionV3 truncated at `Conv2d_4a_3x3`, ResNet-50 truncated at
   `layer1.0`, or the ON-OFF front-end of Berardino et al.) followed by a
   per-neuron Gaussian spatial-attention readout, fitted to the trial-averaged
   responses of the 50 most reliable V1 or V4 channels of the THINGS
   Ventral-stream Spiking Dataset (Papale et al., 2025).
2. **Synthetic responses** (`multigeom.diffusion.responses`). The encoder
   f_φ is evaluated on every TinyImageNet training image; these 50-d
   vectors condition the diffusion model.
3. **Conditional diffusion model** (`multigeom.diffusion.train`). A
   `UNet2DConditionModel` (64×64 grayscale, ~68M parameters) conditioned on
   the responses through a 3-layer MLP and cross-attention. Training follows the
   Gaussian observation model p(r | x) = N(f_φ(x), σ² I)
   with σ = 0.22, and uses classifier-free-guidance dropout (rate 0.1) so
   that the same network also gives the unconditional denoiser. 200 epochs.
4. **G(x) and J(x)** (`multigeom.pipeline.per_image`). The top-50 eigenpairs of
   G(x) from the conditional-pair estimator (Algorithm 1 of the paper,
   `multigeom.geometry.estimator`), and of J(x) = σ⁻² J_fᵀ J_f by
   autograd (`multigeom.geometry.fisher`), on 200 held-out TinyImageNet validation images.
5. **Cross-model comparison** (`multigeom.pipeline.alignment`). Subspace
   alignment A_k = (1/k) ‖V_Aᵀ V_B‖²_F and Bures-Wasserstein
   distance between every pair of (encoder, area) configurations, for
   k = 1, …, 50, averaged over images.
6. **Local metric field** (`multigeom.pipeline.metric_field`). The 2×2 projection of
   G and J onto the leading two eigenvectors at a reference image, on a grid
   of images perturbed along those eigenvectors (Fig. 5).

Figures and the encoder table are drawn from these outputs by the modules in
`multigeom.viz`. Supporting analyses quoted in the paper's appendices (calibration
of σ, Monte-Carlo variance of the conditional-pair versus the
unconditional-conditional estimator, backbone layer sweep) are in `multigeom.fit` and `multigeom.pipeline.estimator_variance`.

## Installation

The package is `multigeom`. The core install gives encoders, the Fisher metric J(x), the diffusion model, the G(x) estimator,
and the information-geometric analyses; two extras add the rest.

```bash
pip install multigeom                 # core: encoders, J(x), the diffusion model, the G(x) estimator, the analyses
pip install "multigeom[paper]"        # + what reproducing the paper needs: the ON-OFF encoder (plenoptic), the TVSD loaders (h5py)
pip install "multigeom[all]"          # + Jupyter and Modal (serverless GPUs)
```

From a clone, for development and to run the paper's pipeline:

```bash
git clone https://github.com/multigeom/multigeom.git
cd multigeom
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev,docs]"
```

`pyproject.toml` declares the supported dependency ranges. The pipeline and tests were run with Python 3.11 and CUDA 12.6.

```python
import multigeom

images = ...                                                # [N, 1, 64, 64] tensor, values in [-1, 1]
encoder = multigeom.load_encoder("inception", area="V1")    # an encoder of the paper, fitted to macaque V1 (downloaded)
responses = encoder.predict(images)                         # [N, 50] predicted neural responses
multigeom.validate_encoder(encoder)                         # checks the encoder contract

image = images[:1]                                          # one image, [1, 1, 64, 64]
J = multigeom.compute_fisher(image, encoder, sigma=0.22)    # eigenpairs of the Fisher metric J(x)

# G(x) also needs the diffusion model `unet` trained for this encoder and its noise schedule `alphas_cumprod`:
G = multigeom.compute_g(image=image, encoder=encoder, unet=unet, alphas_cumprod=a, cfg=cfg, device=device)
```

The command line `multigeom` runs the pipeline: `multigeom jacobian | compute-g | train-diffusion | align`
(`multigeom --help`). The notebooks are in the companion repository [`multigeom-tutorials`](https://github.com/multigeom/multigeom-tutorials); `0_bring_your_own_encoder.ipynb` computes J(x) for a model of your own.

## Data

The three datasets (TVSD, THINGS images, TinyImageNet) and their expected layout are described in
[`multigeom-tutorials`](https://github.com/multigeom/multigeom-tutorials/blob/master/data/README.md), where the notebooks read them. Pretrained backbone weights are
downloaded automatically.

## Running

One (encoder, area) configuration end to end (stages 1-4). `ENCODER` is `inception`, `resnet50` or `onoff`; `AREA` is `V1` or `V4`:

```bash
ENCODER=inception; AREA=V1; RUN=runs; DEVICE=cuda:0

python -m multigeom.fit.train_encoder --config $ENCODER --area $AREA --device $DEVICE \
    --out-dir $RUN/encoders/$ENCODER/$AREA
python -m multigeom.diffusion.responses --config $ENCODER --area $AREA --device $DEVICE \
    --encoder-ckpt $RUN/encoders/$ENCODER/$AREA/best_model.pt --out-dir $RUN/responses/$ENCODER/$AREA
python -m multigeom.diffusion.train --config diffusion --device $DEVICE \
    --responses $RUN/responses/$ENCODER/$AREA/train_responses.pt --out-dir $RUN/diffusion/$ENCODER/$AREA
python -m multigeom.pipeline.per_image --config estimator --encoder-cfg $ENCODER --area $AREA --device $DEVICE \
    --encoder-ckpt $RUN/encoders/$ENCODER/$AREA/best_model.pt \
    --diffusion-ckpt $RUN/diffusion/$ENCODER/$AREA/final_model.pt --out-dir $RUN/per_image/$ENCODER/$AREA
```

Run these for all six configurations of the paper (about 30 A100-hours each, almost all of it diffusion training),
then the comparisons and figures:

```bash
python -m multigeom.pipeline.alignment --runs-dir runs --out-dir runs/metrics
python -m multigeom.pipeline.metric_field --img-idx 0 --out-dir runs/metric_field/img0000
python -m multigeom.viz.table1    --runs-dir runs --out-dir figures/out
python -m multigeom.viz.fig3      --runs-dir runs --out-dir figures/out --img-idx 0
python -m multigeom.viz.fig4      --runs-dir runs --out-dir figures/out --img-idx 0
python -m multigeom.viz.fig5      --runs-dir runs --out-dir figures/out --img-idx 0
python -m multigeom.viz.fig_app_bw --runs-dir runs --out-dir figures/out
```

`multigeom.pipeline.per_image` accepts `--start/--end` to split the 200 test images across GPUs.
Each image is seeded independently (`seed + image index`), so the split does
not change the results. Every checkpoint and output file stores a provenance
record (git commit, command line, seed, config files, package versions).

## Where each result comes from

| Paper item | Script | Inputs |
|---|---|---|
| Table 1 (encoder performance) | `multigeom.viz.table1` | `runs/encoders/*/*/best_model.pt` |
| Fig. 3B-C (measured vs predicted responses) | `multigeom.viz.fig3` | stage 1 |
| Fig. 3D (receptive fields) | `multigeom.viz.fig3` | stage 1 readout parameters |
| Fig. 3E-F (eigenvectors of G and J) | `multigeom.viz.fig3` | stage 4 |
| Fig. 4A (performance of the three encoders) | `multigeom.viz.fig4` | stage 1 |
| Fig. 4B-C (eigenvectors per encoder) | `multigeom.viz.fig4` | stage 4 |
| Fig. 4D-E (subspace alignment vs k) | `multigeom.viz.fig4` | stage 5 |
| Fig. 5 (local metric field) | `multigeom.viz.fig5` | stage 6 |
| Bures-Wasserstein figure (appendix) | `multigeom.viz.fig_app_bw` | stage 5 |
| σ = 0.22 calibration (appendix) | `multigeom.fit.calibrate_sigma` | TVSD |
| CC vs UC estimator variance (appendix) | `multigeom.pipeline.estimator_variance` | stages 1, 3 |
| Backbone layer choice (appendix) | `multigeom.fit.layer_sweep` | TVSD, THINGS |

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

Any fitted PyTorch model can be used as the encoder f(x), if it follows these rules. `multigeom.encoder`
checks them (`multigeom.validate_encoder(encoder)`; `multigeom.compute_fisher` does it automatically for models that
do not use the reference-encoder internals, and `multigeom.compute_g(..., validate=True)` on request) and
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

`2_subspace_alignment.ipynb` (in `multigeom-tutorials`) estimates G(x) and J(x) for many images and configurations and
plots Fig. 4D-E. The images are independent, so they are split over workers (local GPUs, or serverless
GPUs on [Modal](https://modal.com)) with identical results; no DDP/FSDP is involved. All supporting
functions are in `multigeom.pipeline.distribute`. The notebook sets five parameters, `CONFIGS` (one or more
(encoder, area) pairs), `N_IMAGES`, `BACKEND` (`"local"` or `"modal"`), `DEVICE` (`"gpu"` or `"cpu"`) and
`NUM_WORKERS`, and a table in the notebook gives example values for four typical cases. The Modal route
needs `pip install modal` and `modal token new`, and costs money: the notebook prints an estimate and
refuses to launch above `MAX_COST_USD`.

## Local metric field (Fig. 5)

`3_local_metric_field.ipynb` (in `multigeom-tutorials`) plots the local metric field projected onto the leading eigenvectors of
G(x): a reference image is perturbed along the two leading eigenvectors of the Inception model's G(x), and the
metrics G(x) and J(x) of the Inception and ResNet-50 models are drawn as ellipses on that 2-D grid, for V1 and V4.
One GPU is enough: every grid point of every model needs one G(x) estimate (about 15 s on an RTX 5090), so the 5 x 5
grid with both areas takes about 26 minutes and a 3 x 3 grid with one area about 5 minutes. The computation is
`multigeom.pipeline.metric_field`, run as a background process by `multigeom.pipeline.metric_field_job`, and it can be stopped
and resumed. The notebook perturbs by `0.5*sqrt(lambda)` for G(x) (about 0.3) but by a fixed 0.3 for J(x): J has far
larger eigenvalues, and with the same rule its perturbed images are clipped and the ellipses of the two models
swap sizes, unlike the paper's panel B. The paper does not state its step sizes, so this is a choice (the
`metric_field` module itself uses `0.5*sqrt(lambda)` for both metrics, and the notebook runs it once per metric). The paper does not state
its reference image either; the notebook uses validation image 64, the one that best reproduces the pattern of the
paper's panel B among the 200 test images (a scan of the 200 test images with six perturbation settings), whereas the `metric_field` module defaults to image 0. The elongated V1 ellipses
of the paper's panel A are not reproduced (the leading eigenvalues of G(x) are close for every image).

## Supporting analyses

Three modules reproduce numbers quoted in the appendices. They are not part of the main pipeline and write JSON files
with a provenance record. Each module's docstring gives the method.

| Module | Paper statement | Inputs | Cost |
|---|---|---|---|
| `multigeom.fit.calibrate_sigma` | σ = 0.22 (V1), 0.25 (V4) | TVSD files | seconds, CPU |
| `multigeom.pipeline.estimator_variance` | the conditional-pair estimator has 80-95% lower Monte-Carlo variance than the unconditional-conditional form | encoder and diffusion checkpoints, test images | minutes per image, GPU |
| `multigeom.fit.layer_sweep` | `Conv2d_4a_3x3` maximises the test correlation in V1 and V4 | TVSD files, THINGS images | one encoder fit per layer, GPU |

```bash
python -m multigeom.fit.calibrate_sigma --out runs/analysis/sigma_calibration.json
python -m multigeom.pipeline.estimator_variance --encoder-cfg inception \
    --encoder-ckpt runs/encoders/inception/V1/best_model.pt \
    --diffusion-ckpt runs/diffusion/inception/V1/final_model.pt --n-images 6 --n-runs 6 --n-mc 100
python -m multigeom.fit.layer_sweep --backbone inception --area V1
```

On the released TVSD file, the calibration gives 0.2275 (V1) and 0.2451 (V4) on the top-50 channels, and 0.2517 (V1) and
0.2591 (V4) on all channels of the area.

## Tests

```bash
pytest tests/unit -m "not slow and not gpu and not pretrained"   # unit tests: synthetic data, no checkpoints (what GitHub runs)
pytest tests/unit                                                # all unit tests, including the ones that need checkpoints
```

The tests that run the notebooks and compare their figures with stored references are in `multigeom-tutorials`.

## Checkpoints

Trained encoder and diffusion checkpoints, and the per-image G(x)/J(x)
eigendecompositions behind the figures, will be released with a persistent
identifier; this section will link to them.

## Repository layout

```
multigeom/
    __init__.py     public API: load_encoder, NeuralEncoder, compute_fisher, compute_g, subspace_alignment, ...
    encoder.py      the encoder contract (Protocol and validate_encoder)
    encoders/       core: NeuralEncoder, load_encoder, validate_encoder, readouts, the reference encoders (ON-OFF needs the [paper] extra)
    geometry/       core: fisher (compute_fisher), estimator (compute_g), metric_field
    analysis/       core: subspace_alignment, bures_wasserstein
    diffusion/      core: model, config, responses, train
    pipeline/       core: per_image, alignment, metric_field, distribute (local and Modal workers)
    data/           TinyImageNet loader and the canonical image transform
    fit/            extra [paper]: fitting encoders to the TVSD recordings, layer sweep, noise calibration
    viz/            the figures of the paper
    configs/        packaged YAML settings (encoders, diffusion, estimator)
    cli.py  config.py  utils.py
tests/unit/   unit tests (synthetic data)
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
