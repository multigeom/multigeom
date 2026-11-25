"""
The envelope spectrum ρ^env(τ): the error of the unconditional denoiser per unit of
noise variance, against τ = log t,

    ρ^env(τ) = E[(1/t) ‖x̂(x_t) − x‖²],   x ~ p(x),   x_t = x + √t ε,

with x̂(x_t) = E[x | x_t] the posterior mean given by the null token of the diffusion
model. It is an upper bound on the information density ρ(τ) that the responses
carry at each scale (whose integral ½ ∫ ρ dτ is I(R; X)), and the mean over images
of the trace of the stimulus features M_t^env(x) (`multigeom.geometry.env`).

The coding spectrum (`compute_coding_spectrum`) is what the responses take off that
error,

    ρ(τ) = ρ^env(τ) − E[(1/t) ‖x̂(x_t, r) − x‖²],   r ~ p(r | x),

from stimulus–response pairs (x, r = f(x) + σξ) and the conditional posterior mean
x̂(x_t, r), so no approximation of p(r | x_t) is needed: 0 ≤ ρ ≤ ρ^env, and
½ ∫ ρ dτ = I(R; X) in nats, for exact posterior means.

Since x̂ − x = √t (ε − ε̂) by Tweedie's formula, (1/t) ‖x̂ − x‖² is the error of the
noise prediction, ‖ε̂ − ε‖² summed over the pixels: the training loss of the network
at that noise level. Both spectra are computed that way, which avoids subtracting two
nearly equal images at fine noise levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence

import torch

from .estimator import _build_log_tau_grid
from .noise import GaussianNoise, PoissonNoise
from .methods import ConditionalPair


@dataclass
class SpectrumOutput:
    """
    The envelope spectrum of a set of images, returned by `compute_env_spectrum`:
    log τ_j [J] and, for each image, its error per unit of noise variance at each
    level, `values` [n_images, J]; their mean over the images is ρ^env(τ) (`mean`).
    """

    log_tau: torch.Tensor  # [J]            log of the noise variance τ_j
    values: torch.Tensor  # [n_images, J]  E_{x_t|x}[(1/t) ‖x̂(x_t) − x‖²] per image

    @property
    def mean(self) -> torch.Tensor:
        """ρ^env(τ) [J]: the mean of the curves of the images."""
        return self.values.mean(dim=0)


def _level_errors(
    pairs: torch.Tensor,
    unet: torch.nn.Module,
    alphas_cumprod: torch.Tensor,
    t_j: int,
    batch_size: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """
    ‖ε̂(x_t) − ε‖² = (1/t) ‖x̂(x_t) − x‖² for each clean image of `pairs`
    ([n, 1, H, W], on the device of the network) noised at the DDPM step `t_j`, with the
    null token: [n] on the CPU.
    """
    # the weights of the clean image and of the noise at this level
    sqrt_ab = float(alphas_cumprod[t_j].sqrt())
    sqrt_1m = float((1.0 - alphas_cumprod[t_j]).sqrt())
    device = pairs.device

    # the images in batches: noise them, predict the noise, keep the squared error
    errors = []
    for clean in pairs.split(batch_size):
        noise = torch.randn(clean.shape, device=device, generator=generator)
        steps = torch.full((len(clean),), t_j, device=device, dtype=torch.long)
        null = torch.zeros(len(clean), unet.n_neurons, device=device)
        predicted = unet(sqrt_ab * clean + sqrt_1m * noise, steps, null)
        errors.append((predicted - noise).flatten(1).pow(2).sum(dim=1).cpu())
    return torch.cat(errors)


@torch.no_grad()
def estimate_env_spectrum(
    images: torch.Tensor,
    unet: torch.nn.Module,
    alphas_cumprod: torch.Tensor,
    timesteps: Sequence[int],
    device: torch.device,
    n_noise: int = 8,
    batch_size: int = 400,
    generator: torch.Generator | None = None,
) -> SpectrumOutput:
    """
    Compute the envelope spectrum of `images` ([n, 1, H, W] in [-1, 1]) at the DDPM
    steps `timesteps` (any increasing steps, not only the grid of the estimator
    config): for each image and step, the mean over `n_noise` noisy versions of
    (1/t) ‖x̂(x_t) − x‖², with x̂ the posterior mean of the null token of `unet` (see
    the module). The network sees `batch_size` noisy images at a time; `generator` (on
    `device`) seeds the noise.

    Returns:
        SpectrumOutput with `log_tau` [J] and `values` [n, J]; `mean` is ρ^env(τ).
    """
    # the arguments
    if images.dim() != 4:
        raise ValueError(f"Expected images [n, C, H, W], got {tuple(images.shape)}")
    if n_noise < 1 or batch_size < 1:
        raise ValueError(
            f"n_noise and batch_size must be at least 1, got {n_noise}, {batch_size}"
        )

    # inference mode, and each image repeated once per noisy version
    unet.eval()
    pairs = images.to(device).repeat_interleave(n_noise, dim=0)

    # the errors of every noisy version at every level [J, n · n_noise]
    errors = torch.stack(
        [
            _level_errors(pairs, unet, alphas_cumprod, t_j, batch_size, generator)
            for t_j in timesteps
        ]
    )

    # the mean over the noisy versions of each image, [n, J], and the noise levels
    values = errors.view(len(timesteps), len(images), n_noise).mean(dim=2).T
    log_tau = _build_log_tau_grid(timesteps, alphas_cumprod).cpu()
    return SpectrumOutput(log_tau=log_tau, values=values)


@dataclass
class CodingSpectrumOutput:
    """
    The coding spectrum of a set of images, returned by `compute_coding_spectrum`:
    log τ_j [J] and, for each image and level [n_images, J], four means over its noisy
    versions, each with a response r = f(x) + σξ to the image itself (a
    stimulus–response pair): `env` = (1/t) ‖x̂(x_t) − x‖², the error without the
    responses (ρ^env); `coded` = (1/t) ‖x̂(x_t, r) − x‖², the error with the response;
    `gain` = (1/t) ‖x̂(x_t, r) − x̂(x_t)‖², what the response changes; and `sampled` =
    ½ (1/t) ‖x̂(x_t, r₁) − x̂(x_t, r₂)‖², with r₁ and r₂ drawn given the denoised
    image, the trace of what `compute_g_per_scale` estimates. ρ(τ) is the mean of
    env − coded (`rho`, the note's eq. 7); for exact posterior means it is also the
    mean of gain (`rho_gain`), and `rho_sampled` is the mean of sampled, which
    approximates it.
    """

    log_tau: torch.Tensor  # [J]            log of the noise variance τ_j
    env: torch.Tensor  # [n_images, J]      the error without the responses
    coded: torch.Tensor  # [n_images, J]    the error with a response to the image
    gain: torch.Tensor  # [n_images, J]     what that response changes
    sampled: torch.Tensor  # [n_images, J]  half the squared change between two draws

    @property
    def rho(self) -> torch.Tensor:
        """ρ(τ) [J]: the error without the responses minus the error with them."""
        return (self.env - self.coded).mean(dim=0)

    @property
    def rho_gain(self) -> torch.Tensor:
        """ρ(τ) [J] from what the responses change, equal to `rho` for exact means."""
        return self.gain.mean(dim=0)

    @property
    def rho_sampled(self) -> torch.Tensor:
        """[J] the curve of `compute_g_per_scale`, with responses drawn given x̂."""
        return self.sampled.mean(dim=0)


def _squared_norms(rows: torch.Tensor) -> torch.Tensor:
    """The squared norm of each row of `rows` [B, ...], on the CPU: [B]."""
    return rows.flatten(1).pow(2).sum(dim=1).cpu()


def _level_terms(
    pairs: torch.Tensor,
    means: torch.Tensor,
    unet: torch.nn.Module,
    sampler: ConditionalPair,
    alphas_cumprod: torch.Tensor,
    t_j: int,
    response_noise: GaussianNoise | PoissonNoise,
    batch_size: int,
    generator: torch.Generator | None,
) -> dict:
    """
    The four terms of `CodingSpectrumOutput` for each clean image of `pairs`
    ([n, 1, H, W], with the mean responses `means` [n, N] to them) noised at the DDPM
    step `t_j`: {name: [n] on the CPU}.
    """
    # the weights of the clean image and of the noise at this level
    sqrt_ab = float(alphas_cumprod[t_j].sqrt())
    sqrt_1m = float((1.0 - alphas_cumprod[t_j]).sqrt())
    device = pairs.device
    terms = {"env": [], "coded": [], "gain": [], "sampled": []}

    for clean, mean in zip(pairs.split(batch_size), means.split(batch_size)):
        # the noisy images, and a response to each clean image
        noise = torch.randn(clean.shape, device=device, generator=generator)
        x_t = sqrt_ab * clean + sqrt_1m * noise
        steps = torch.full((len(clean),), t_j, device=device, dtype=torch.long)
        responses = response_noise.sample(mean, generator)

        # the noise predictions without and with that response
        null = torch.zeros(len(clean), unet.n_neurons, device=device)
        eps_null = unet(x_t, steps, null)
        eps_coded = unet(x_t, steps, responses)

        # the errors, what the response changes, and the pair drawn given x̂
        terms["env"].append(_squared_norms(eps_null - noise))
        terms["coded"].append(_squared_norms(eps_coded - noise))
        terms["gain"].append(_squared_norms(eps_coded - eps_null))
        terms["sampled"].append(
            0.5 * _squared_norms(sampler.sample_rows(x_t, steps, t_j))
        )
    return {name: torch.cat(values) for name, values in terms.items()}


@torch.no_grad()
def estimate_coding_spectrum(
    images: torch.Tensor,
    encoder: torch.nn.Module,
    unet: torch.nn.Module,
    alphas_cumprod: torch.Tensor,
    timesteps: Sequence[int],
    device: torch.device,
    sigma: float,
    n_noise: int = 8,
    batch_size: int = 400,
    generator: torch.Generator | None = None,
    noise: GaussianNoise | PoissonNoise | None = None,
) -> CodingSpectrumOutput:
    """
    Compute the coding spectrum of `images` ([n, 1, H, W] in [-1, 1]) at the DDPM steps
    `timesteps`, for the encoder `encoder` (its responses r = f(x) + σξ with σ =
    `sigma`, which must be that of the training of `unet`) and its conditional
    diffusion model `unet`: for each image and step, the four terms of
    `CodingSpectrumOutput`, each the mean over `n_noise` noisy versions, with a new
    response for each. `batch_size` and `generator` as in `compute_env_spectrum`.
    `noise` replaces the Gaussian noise of σ by another noise of the responses
    (`multigeom.geometry.noise`, e.g. Poisson for the retina models).

    Returns:
        CodingSpectrumOutput; its `rho` is ρ(τ), the note's eq. 7.
    """
    # the arguments
    if images.dim() != 4:
        raise ValueError(f"Expected images [n, C, H, W], got {tuple(images.shape)}")
    if n_noise < 1 or batch_size < 1:
        raise ValueError(f"n_noise, batch_size must be >= 1: {n_noise}, {batch_size}")

    # inference mode, and the mean response to each clean image, once
    encoder.eval()
    unet.eval()
    images = images.to(device)
    means = torch.cat([encoder(batch) for batch in images.split(batch_size)])

    # each image and its mean response repeated once per noisy version
    pairs = images.repeat_interleave(n_noise, dim=0)
    pair_means = means.repeat_interleave(n_noise, dim=0)

    # the noise of the responses, and the responses drawn given the denoised image, as
    # `compute_g_per_scale` draws them
    noise = noise or GaussianNoise(sigma)
    settings = SimpleNamespace(
        monte_carlo=SimpleNamespace(observation_noise_sigma=sigma, noise=noise)
    )
    sampler = ConditionalPair(
        image=images[:1],
        encoder=encoder,
        model=unet,
        alphas_cumprod=alphas_cumprod,
        cfg=settings,
        device=device,
        generator=generator,
    )

    # the four terms at every level, then their means over the noisy versions [n, J]
    levels = [
        _level_terms(
            pairs,
            pair_means,
            unet,
            sampler,
            alphas_cumprod,
            t_j,
            noise,
            batch_size,
            generator,
        )
        for t_j in timesteps
    ]
    terms = {
        name: torch.stack([level[name] for level in levels])
        .view(len(timesteps), len(images), n_noise)
        .mean(dim=2)
        .T
        for name in ("env", "coded", "gain", "sampled")
    }
    log_tau = _build_log_tau_grid(timesteps, alphas_cumprod).cpu()
    return CodingSpectrumOutput(log_tau=log_tau, **terms)


def compute_prior_variances(images_per_class: int) -> torch.Tensor:
    """
    The variances of the TinyImageNet training images along their principal components,
    largest first, [P] in float64: the eigenvalues of the pixel covariance of the first
    `images_per_class` images of each of the 200 classes.
    """
    from ..datasets import CANONICAL_TRANSFORM, TinyImageNetDataset

    # the training images, one row of pixels each
    train = TinyImageNetDataset(
        split="train", transform=CANONICAL_TRANSFORM, images_per_class=images_per_class
    )
    pixels = torch.stack([train[index] for index in range(len(train))]).flatten(1)

    # the eigenvalues of their covariance, largest first (rounding can make 0 negative)
    covariance = torch.cov(pixels.double().T)
    return torch.linalg.eigvalsh(covariance).flip(0).clamp(min=0)


def compute_gaussian_env(variances: torch.Tensor, log_t: torch.Tensor) -> torch.Tensor:
    """
    The envelope spectrum of a Gaussian prior with these `variances`, which is also the
    error of the best linear denoiser (the Wiener filter) for any prior with them:
    Σ_i c_i / (c_i + t) at each log t [J]. It bounds the true curve from above, and
    tends to the number of pixels at fine levels and to Σ_i c_i / t at coarse ones.
    """
    # the noise variances, then the sum over the principal components
    noise_variances = log_t.double().exp()[:, None]
    return (variances[None, :] / (variances[None, :] + noise_variances)).sum(dim=1)
