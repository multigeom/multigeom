"""
The retina models of open-retina's information geometry (Lorenzi adaptation cells):
the mosaic of one cell type and its conditional diffusion model, loaded so that the
estimators of `multigeom.geometry` run on them. Needs the `openretina` package with its
`insilico.information_geometry` module (Baptiste Lorenzi's, not released: installed
from the private repository steevelaquitaine/open-retina-multigeom), imported when a
function is called.

    load_retina(area, runs_dir, frames_root, device)
        the mosaic encoder (its output is the Poisson rate λ of each cell), the
        diffusion model, the noise schedule, the Poisson noise of the responses, and
        the map between the model's and the diffusion model's image units
    load_retina_test_images(models, frames_root, n)
        the test images of open-retina's `run_lorenzi` (TinyImageNet training images
        50,000 onwards, in its shuffled order), in the diffusion model's units

The checkpoints are laid out as those of V1 and V4 (the Hugging Face repository
steevelaquitaine/multigeom, downloaded into runs/), the Lorenzi adaptation model as the
encoder and the cell type and cell as the area, e.g. "retina-W3_176-exp7":

    runs/encoders/lorenzi/<area>/best_model.pt     the cell's Lorenzi model
    runs/diffusion/lorenzi/<area>/final_model.pt   its mosaic's conditional diffusion
                                                   model, trained on TinyImageNet

The images are 120 x 120 with one channel; the mosaic is the cell's readout slid over a
hexagonal lattice 200 µm apart, as the diffusion models were trained.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from ..geometry.noise import PoissonNoise

# the canvas of the diffusion models, the spacing of the mosaic, and the training
# images the image units were fitted to (open-retina's run_lorenzi)
CANVAS, SPACING_UM, N_TRAIN_FRAMES = 120, 200.0, 15000

# the encoder of the checkpoints' layout (runs/<stage>/<encoder>/<area>/)
ENCODER = "lorenzi"


class RateEncoder(nn.Module):
    """The mosaic encoder of open-retina: the rates λ [B, N], not standardised."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    @property
    def n_neurons(self) -> int:
        """The number of cells of the mosaic."""
        return self.encoder.n_neurons

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The rates [B, N] of images [B, 1, 120, 120] in diffusion-model units."""
        return self.encoder.forward_raw(x)


def use_math_attention(device: torch.device) -> None:
    """
    On GPUs of compute capability 10 and above (Blackwell, e.g. RTX 5090), use the math
    kernel of attention: the fast kernels return zeros there, which makes G(x) vanish
    (as in open-retina's run_lorenzi). Changes a global setting of torch.
    """
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 10:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def compute_normalizer(frames_root: Path, cache: Path | None = None):
    """
    The map between the encoder's and the diffusion model's image units, fitted to the
    first 15,000 shuffled TinyImageNet training images at 120 x 120 (as in training),
    read from `cache` if it exists, else computed and stored there.
    """
    from openretina.insilico.information_geometry.stimuli import (
        StimulusNormalizer,
        load_tinyimagenet_frames,
    )

    # stored: read it
    if cache is not None and Path(cache).exists():
        return StimulusNormalizer.from_state_dict(torch.load(cache))

    # else from the training images, then stored
    frames = load_tinyimagenet_frames(N_TRAIN_FRAMES, resize=CANVAS, root=frames_root)
    normalizer = StimulusNormalizer.from_frames(frames)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        torch.save(normalizer.state_dict(), cache)
    return normalizer


def load_mosaic(encoder_file: Path, normalizer, device: torch.device):
    """
    The mosaic of a cell: its Lorenzi model (`encoder_file`) with its readout slid over
    a hexagonal lattice 200 µm apart on the 120 x 120 canvas, as open-retina's
    `build_sliding_readout_mosaic_encoder`.
    """
    from openretina.insilico.information_geometry.retina_encoder import (
        build_sliding_readout_mosaic_encoder,
    )
    from openretina.insilico.information_geometry.run_lorenzi import load_lorenzi
    from openretina.insilico.information_geometry.stimuli import PadSpec

    # the cell's model, and the spacing of the lattice in pixels of its core's output
    model, session, _ = load_lorenzi(str(encoder_file), device)
    stride = getattr(model.core.subunit, "stride", 1)
    stride = stride[0] if isinstance(stride, (tuple, list)) else stride
    spacing = SPACING_UM / (float(stride) * model.data_info["microns_per_px"])

    # the mosaic on the canvas
    pad = PadSpec.to_multiple(CANVAS, CANVAS, 4)
    mosaic = build_sliding_readout_mosaic_encoder(
        model,
        session,
        0,
        normalizer,
        pad,
        device=str(device),
        layout="hex",
        nn_spacing_px=spacing,
    )
    return mosaic.to(device), pad


def load_retina(
    area: str,
    runs_dir: Path,
    frames_root: Path,
    device: torch.device,
    normalizer_cache: Path | None = None,
) -> SimpleNamespace:
    """
    The retina models of `area` (e.g. "retina-W3_176-exp7"), from `runs_dir`:
    `encoder` (RateEncoder), `unet` (open-retina's NeuralConditionedUNet),
    `alphas_cumprod`, `noise` (PoissonNoise), `normalizer`, `pad` and `cell`.
    `frames_root` holds the TinyImageNet training images.
    """
    from openretina.insilico.information_geometry.diffusion import (
        NeuralConditionedUNet,
        build_scheduler,
    )

    # the checkpoint, the image units, and the mosaic of its cell
    use_math_attention(device)
    diffusion_file = Path(runs_dir) / "diffusion" / ENCODER / area / "final_model.pt"
    stored = torch.load(diffusion_file, map_location="cpu", weights_only=False)
    normalizer = compute_normalizer(frames_root, normalizer_cache)
    encoder_file = Path(runs_dir) / "encoders" / ENCODER / area / "best_model.pt"
    mosaic, pad = load_mosaic(encoder_file, normalizer, device)

    # the diffusion model, conditioned on the responses of the mosaic's cells
    unet = NeuralConditionedUNet(
        in_channels=1, sample_size=(pad.hp, pad.wp), n_neurons=mosaic.n_neurons
    )
    unet.load_state_dict(stored["unet"], strict=True)

    # the noise schedule, and the Poisson noise of the responses
    alphas_cumprod = build_scheduler().alphas_cumprod.to(device)
    noise = PoissonNoise(
        stored["poisson_gain"], stored["resp_mean"], stored["resp_std"]
    )
    return SimpleNamespace(
        encoder=RateEncoder(mosaic).eval(),
        unet=unet.to(device).eval(),
        alphas_cumprod=alphas_cumprod,
        noise=noise,
        normalizer=normalizer,
        pad=pad,
        cell=stored["cell"],
    )


def load_retina_test_images(
    models: SimpleNamespace, frames_root: Path, n: int = 6
) -> torch.Tensor:
    """
    The test images of open-retina's run_lorenzi, [n, 1, 120, 120] in the diffusion
    model's units: TinyImageNet training images 50,000 to 50,000 + n in its shuffled
    order, standardised together (so `n` changes them), as it did.
    """
    from openretina.insilico.information_geometry.stimuli import (
        load_tinyimagenet_frames,
    )

    # the images, then in the diffusion model's units on its canvas
    frames = load_tinyimagenet_frames(n, resize=CANVAS, root=frames_root, offset=50000)
    return models.pad.pad(models.normalizer.to_diffusion(frames), value=0.0)


def compute_retina_prior_variances(
    models: SimpleNamespace, frames_root: Path, device: torch.device
) -> torch.Tensor:
    """
    The variances of the images the retina diffusion models were trained on, along
    their principal components, largest first, [P] in float64: the eigenvalues of the
    pixel covariance of the first 15,000 shuffled TinyImageNet training images at
    120 x 120, in the diffusion model's units (the images its normalizer was fitted
    to). Computed on `device` in float32 (the covariance is 14,400 x 14,400).
    """
    from openretina.insilico.information_geometry.stimuli import (
        load_tinyimagenet_frames,
    )

    # the training images in the diffusion model's units, one row of pixels each
    frames = load_tinyimagenet_frames(N_TRAIN_FRAMES, resize=CANVAS, root=frames_root)
    pixels = models.pad.pad(models.normalizer.to_diffusion(frames), value=0.0)
    pixels = pixels.flatten(1).to(device)

    # the eigenvalues of their covariance, largest first (rounding can make 0 negative)
    pixels = pixels - pixels.mean(dim=0)
    covariance = pixels.T @ pixels / (len(pixels) - 1)
    variances = torch.linalg.eigvalsh(covariance).flip(0).clamp(min=0)
    return variances.double().cpu()


def load_stimulus_frames(files: list, size: int = CANVAS) -> torch.Tensor:
    """
    The image files `files` prepared as open-retina prepares its stimuli, [n, 1, size,
    size] in the units of the recorded stimuli (and of the retina encoders' inputs):
    grayscale, resized (bilinear), standardised together to mean 0 and std 1, and
    clipped to [-2.4, 2.5].
    """
    import numpy as np
    from PIL import Image

    # grayscale, resized, in [0, 1]
    frames = torch.stack(
        [
            torch.from_numpy(
                np.asarray(
                    Image.open(f).convert("L").resize((size, size), Image.BILINEAR),
                    dtype=np.float32,
                )
                / 255.0
            )[None]
            for f in files
        ]
    )

    # standardised together, and clipped
    frames = (frames - frames.mean()) / frames.std().clamp(min=1e-6)
    return frames.clamp(-2.4, 2.5)


def load_retina_images(models: SimpleNamespace, files: list) -> torch.Tensor:
    """
    The image files `files` prepared as open-retina prepares its images, [n, 1, 120,
    120] in the diffusion model's units: `load_stimulus_frames`, then mapped by the
    normalizer of `models` and padded.
    """
    frames = load_stimulus_frames(files)
    return models.pad.pad(models.normalizer.to_diffusion(frames), value=0.0)
