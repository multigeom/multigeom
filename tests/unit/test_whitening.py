"""
Tests for multigeom.datasets.whitening: the whitening filter of Olshausen & Field,
K(f) = f exp(−(f / f₀)⁴).

Analytic references:
  - A Gaussian random field with power spectrum A / f^α (drawn by shaping white noise
    in the Fourier domain) has, by construction, the exponent α.
  - Filtered by K, a 1/f² field has the spectrum K² / f² ∝ exp(−2 (f / f₀)⁴), flat to
    within 1% below 0.3 f₀.
"""

from __future__ import annotations

import torch

from multigeom.datasets.whitening import (
    WhiteningFilter,
    compute_filter_gain,
    compute_radial_frequencies,
    estimate_power_spectrum,
    fit_power_law,
)


def make_power_law_field(n: int, size: int, exponent: float, seed: int = 0):
    """`n` periodic Gaussian fields of `size` x `size` pixels whose power spectrum is
    1 / f^exponent, [n, 1, size, size] in float64."""
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(n, 1, size, size, generator=generator, dtype=torch.float64)
    radius = compute_radial_frequencies(size)
    amplitude = radius.clamp(min=1.0 / size) ** (-exponent / 2)
    amplitude[0, 0] = 0.0
    return torch.fft.ifft2(torch.fft.fft2(noise) * amplitude).real


def test_fit_power_law_recovers_the_exponent_of_a_power_law_field():
    """Verify that the exponent fitted to the spectrum of 1/f² fields is 2 within
    0.1."""
    fields = make_power_law_field(400, 64, 2.0)
    freqs, power = estimate_power_spectrum(fields)
    _, exponent = fit_power_law(freqs, power, (0.04, 0.25))
    assert abs(exponent - 2.0) < 0.1


def test_filter_gain_is_one_at_its_peak_and_zero_at_dc():
    """Verify that the gain of the filter is 0 at zero frequency, largest and equal to 1
    at f₀ / 4^¼, and below 2% of its peak at 1.5 f₀ (1.7% exactly)."""
    freqs = torch.linspace(0.0, 0.5, 5001, dtype=torch.float64)
    gain = compute_filter_gain(freqs, 0.3)
    assert gain[0] == 0.0
    peak = 0.3 / 4**0.25
    assert abs(float(freqs[gain.argmax()]) - peak) < 1e-3
    assert abs(float(compute_filter_gain(torch.tensor([peak]), 0.3)) - 1.0) < 1e-12
    assert float(compute_filter_gain(torch.tensor([0.45]), 0.3)) < 0.02


def test_filtered_field_is_white_below_the_cut_off():
    """Verify that a 1/f² field filtered with a cut-off of 0.4 cycles/pixel has a flat
    spectrum, within 10%, from 0.03 to 0.12 cycles/pixel, where the filter is ≈ f."""
    fields = make_power_law_field(400, 64, 2.0)
    freqs, power = estimate_power_spectrum(WhiteningFilter(0.4, pad=0)(fields))
    flat = power[(freqs >= 0.03) & (freqs <= 0.12)]
    assert float(flat.max() / flat.min()) < 1.1


def test_kernel_is_isotropic_with_a_centre_and_a_surround_of_opposite_signs():
    """Verify that the spatial kernel of the filter is symmetric under transposition and
    a 90° rotation, positive at its centre and negative in a ring around it."""
    kernel = WhiteningFilter(0.4).kernel(64)
    assert torch.allclose(kernel[1:, 1:], kernel[1:, 1:].T, atol=1e-12)
    assert torch.allclose(
        kernel[1:, 1:], kernel[1:, 1:].flip(0).T.flip(0).T, atol=1e-12
    )
    assert kernel[32, 32] > 0
    assert kernel[32, 33:38].min() < 0


def test_from_images_keeps_the_pixel_variance_and_the_state_dict_round_trips():
    """Verify that the filter fitted to images keeps their pixel variance, and that the
    filter rebuilt from its state dict filters identically."""
    fields = make_power_law_field(200, 64, 2.0).float()
    whitening = WhiteningFilter.from_images(fields, 0.4)
    filtered = whitening(fields)
    assert abs(float(filtered.var() / fields.var()) - 1.0) < 1e-3
    assert filtered.shape == fields.shape and filtered.dtype == fields.dtype
    rebuilt = WhiteningFilter.from_state_dict(whitening.state_dict())
    assert torch.equal(rebuilt(fields), filtered)


def test_invert_undoes_the_filter_where_it_passes_the_image():
    """
    Verify that un-whitening a whitened 1/f² field gives back the field, mean removed,
    within 2% of its variance: the regularised inverse recovers what the filter passes,
    and only the mean (and the frequencies above the cut-off, nearly empty in a 1/f²
    field) is lost.
    """
    # 1/f² fields, whitened then un-whitened
    fields = make_power_law_field(8, 64, 2.0)
    fields = fields - fields.mean(dim=(-2, -1), keepdim=True)
    whitening = WhiteningFilter.from_images(fields, cutoff=0.4, pad=16)
    restored = whitening.invert(whitening(fields), regularization=1e-4)

    # the error, relative to the variance of the fields
    error = (restored - fields).pow(2).mean() / fields.pow(2).mean()
    assert float(error) < 0.02, float(error)
