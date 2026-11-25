"""
Spatial whitening of images with the filter of Olshausen & Field (1996, 1997).

Natural images have a power spectrum that falls about as 1/f², so their amplitude
spectrum falls as 1/f. The filter K(f) = f exp(−(f / f₀)⁴) multiplies the amplitude by
f, which flattens the spectrum (whitening), and its exponential rolls off sharply near
the cut-off f₀ (0.4 cycles/pixel), where the images carry little signal and much noise
(JPEG, aliasing). Without that roll-off the filter would amplify the noise at the top
of the spectrum and make the images look like speckle; with it, its spatial kernel is a
compact centre–surround, like that of retinal ganglion cells. The filter is 0 at f = 0
(the mean of the image is removed) and largest at f₀ / 4^¼ ≈ 0.71 f₀, where it is
scaled to 1.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_radial_frequencies(size: int, dtype=torch.float64) -> torch.Tensor:
    """The spatial frequency |f| in cycles per pixel of each entry of an FFT of a
    `size` x `size` image, [size, size] in the FFT's layout (0 at [0, 0])."""
    f = torch.fft.fftfreq(size, dtype=dtype)
    return (f[:, None] ** 2 + f[None, :] ** 2).sqrt()


def estimate_power_spectrum(
    images: torch.Tensor, n_bins: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The radially averaged power spectrum of `images` [n, (1,) H, W], H = W: the mean
    over the images of |FFT|² of each image minus its mean, apodised by a 2-D Hann
    window (so the edges of the image do not add a cross of spurious power along the
    axes), per pixel of the window. Returns, for `n_bins` rings of frequency from 0
    to the Nyquist frequency 0.5 cycles/pixel, the mean frequency and the mean power
    of the ring, [n_bins] in float64.
    """
    # the images without their mean, apodised
    images = images.reshape(-1, *images.shape[-2:]).double()
    size = images.shape[-1]
    window = torch.hann_window(size, periodic=False, dtype=torch.float64)
    window = window[:, None] * window[None, :]
    centred = images - images.mean(dim=(-2, -1), keepdim=True)

    # the power at each frequency, averaged over the images, per pixel of the window
    power = torch.zeros(size, size, dtype=torch.float64)
    for batch in centred.split(1000):
        power += (torch.fft.fft2(batch * window).abs() ** 2).sum(dim=0)
    power /= len(images) * window.pow(2).sum()

    # its mean in rings of frequency, up to Nyquist
    radius = compute_radial_frequencies(size)
    edges = torch.linspace(0.0, 0.5, n_bins + 1, dtype=torch.float64)
    index = torch.bucketize(radius, edges, right=True) - 1
    inside = (index >= 0) & (index < n_bins)
    total = torch.zeros(n_bins, dtype=torch.float64).index_add_(
        0, index[inside], power[inside]
    )
    count = torch.zeros(n_bins, dtype=torch.float64).index_add_(
        0, index[inside], torch.ones_like(power[inside])
    )
    freqs = torch.zeros(n_bins, dtype=torch.float64).index_add_(
        0, index[inside], radius[inside]
    )
    return freqs / count.clamp(min=1), total / count.clamp(min=1)


def fit_power_law(
    freqs: torch.Tensor, power: torch.Tensor, band: tuple[float, float]
) -> tuple[float, float]:
    """
    The amplitude A and the exponent α of S(f) = A / f^α, fitted by least squares on a
    log–log scale to the power spectrum (`freqs`, `power`) within `band` (cycles/pixel),
    which should exclude the lowest frequencies (few samples, the window) and the top of
    the spectrum (noise floor).
    """
    keep = (freqs >= band[0]) & (freqs <= band[1]) & (power > 0)
    if int(keep.sum()) < 2:
        raise ValueError(f"Fewer than 2 frequencies in the band {band}")
    x, y = freqs[keep].log(), power[keep].log()
    slope = ((x - x.mean()) * (y - y.mean())).sum() / (x - x.mean()).pow(2).sum()
    intercept = y.mean() - slope * x.mean()
    return float(intercept.exp()), float(-slope)


def compute_filter_gain(freqs: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    The gain K(f) = f exp(−(f / f₀)⁴) of the whitening filter of Olshausen & Field at
    the frequencies `freqs` (cycles/pixel), f₀ = `cutoff`, scaled to 1 at its peak
    f₀ / 4^¼: ≈ f (whitening of a 1/f² spectrum) well below the cut-off, ≈ 0 above it.
    """
    freqs = freqs.double()
    peak = cutoff / 4**0.25
    gain = freqs * torch.exp(-((freqs / cutoff) ** 4))
    return gain / (peak * torch.exp(torch.tensor(-0.25, dtype=torch.float64)))


class WhiteningFilter:
    """
    The whitening filter of Olshausen & Field with cut-off `cutoff` cycles/pixel
    (`compute_filter_gain`), times a global `scale`. Calling it on images [..., H, W]
    pads them by `pad` pixels on each side by reflection (so the filter does not wrap
    one edge of the image onto the other), filters them by FFT, and crops them back.
    """

    def __init__(self, cutoff: float = 0.4, scale: float = 1.0, pad: int = 16):
        self.cutoff = float(cutoff)
        self.scale = float(scale)
        self.pad = int(pad)

    @classmethod
    def from_images(
        cls, images: torch.Tensor, cutoff: float = 0.4, pad: int = 16
    ) -> "WhiteningFilter":
        """
        The filter with cut-off `cutoff`, scaled so that the images `images`
        [n, 1, H, W] filtered have the same pixel variance as before.
        """
        whitening = cls(cutoff, 1.0, pad)
        filtered = torch.cat([whitening(batch) for batch in images.split(1000)])
        whitening.scale = float(
            (images.double().var() / filtered.double().var()).sqrt()
        )
        return whitening

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """The filtered images, the shape and dtype of `images` [..., H, W]."""
        shape, dtype = images.shape, images.dtype
        flat = images.reshape(-1, 1, *shape[-2:]).double()

        # padded by reflection, filtered in the Fourier domain, cropped back
        padded = F.pad(flat, [self.pad] * 4, mode="reflect")
        height, width = padded.shape[-2:]
        fy = torch.fft.fftfreq(height, dtype=torch.float64)
        fx = torch.fft.fftfreq(width, dtype=torch.float64)
        radius = (fy[:, None] ** 2 + fx[None, :] ** 2).sqrt()
        gain = compute_filter_gain(radius, self.cutoff)
        filtered = torch.fft.ifft2(torch.fft.fft2(padded) * gain).real
        rows = slice(self.pad, self.pad + shape[-2])
        columns = slice(self.pad, self.pad + shape[-1])
        return (self.scale * filtered[..., rows, columns]).reshape(shape).to(dtype)

    def invert(
        self, images: torch.Tensor, regularization: float = 0.01
    ) -> torch.Tensor:
        """
        The images [..., H, W] un-whitened, with the regularised (Wiener) inverse of the
        filter, K / (K² + λ) / scale with λ = `regularization` (the gain K is 1 at its
        peak): about 1 / K where the filter passes the image, and 0 where it removed it
        (the mean of the image, at f = 0, cannot be recovered; nor can the frequencies
        above the cut-off). Padded, filtered and cropped as the filter itself.
        """
        shape, dtype = images.shape, images.dtype
        flat = images.reshape(-1, 1, *shape[-2:]).double()

        # padded by reflection, divided by the gain in the Fourier domain, cropped back
        padded = F.pad(flat, [self.pad] * 4, mode="reflect")
        height, width = padded.shape[-2:]
        fy = torch.fft.fftfreq(height, dtype=torch.float64)
        fx = torch.fft.fftfreq(width, dtype=torch.float64)
        radius = (fy[:, None] ** 2 + fx[None, :] ** 2).sqrt()
        gain = compute_filter_gain(radius, self.cutoff)
        inverse = gain / (gain**2 + regularization)
        restored = torch.fft.ifft2(torch.fft.fft2(padded) * inverse).real
        rows = slice(self.pad, self.pad + shape[-2])
        columns = slice(self.pad, self.pad + shape[-1])
        return (restored[..., rows, columns] / self.scale).reshape(shape).to(dtype)

    def kernel(self, size: int = 64) -> torch.Tensor:
        """The spatial kernel of the filter on a `size` x `size` grid, centred,
        [size, size] in float64 (scale included)."""
        gain = compute_filter_gain(compute_radial_frequencies(size), self.cutoff)
        return self.scale * torch.fft.fftshift(torch.fft.ifft2(gain).real)

    def state_dict(self) -> dict:
        """The settings of the filter, to store it (`from_state_dict` rebuilds it)."""
        return {"cutoff": self.cutoff, "scale": self.scale, "pad": self.pad}

    @classmethod
    def from_state_dict(cls, state: dict) -> "WhiteningFilter":
        """The filter stored by `state_dict`."""
        return cls(**state)
