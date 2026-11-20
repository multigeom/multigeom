"""
Check the analyses of the estimates at each noise scale (sign alignment, normalised
curves, information, paired bootstrap), the Gaussian bound of the envelope spectrum and
the Fisher information per neuron on inputs whose answer is known.
"""

from __future__ import annotations

import torch

from multigeom.analysis import align_signs


def test_align_signs_flips_only_signs_to_agree_with_coarser_levels():
    """
    Verify that `align_signs` (`multigeom.analysis`) only flips signs, makes each
    eigenvector agree with the same one at the next coarser level, and gives the
    coarsest a positive sum.
    """
    # random eigenvectors of 6 levels, with random signs
    generator = torch.Generator().manual_seed(0)
    eigvecs = torch.randn(6, 3, 50, generator=generator)
    signs = torch.where(torch.rand(6, 3, 1, generator=generator) < 0.5, -1.0, 1.0)
    aligned = align_signs(eigvecs * signs)

    # only the signs change
    torch.testing.assert_close(aligned.abs(), eigvecs.abs())

    # each eigenvector agrees with the next coarser level; the coarsest sum to > 0
    assert bool(((aligned[:-1] * aligned[1:]).sum(dim=2) >= 0).all())
    assert bool((aligned[-1].sum(dim=1) >= 0).all())


def test_compute_gaussian_env_on_known_variances():
    """
    Verify that `compute_gaussian_env` (`multigeom.geometry`), the envelope spectrum of
    a Gaussian prior, gives the sum over the directions of their variance c_i over c_i
    plus the noise variance t, Σ_i c_i / (c_i + t), on known variances: the number of
    directions that vary at tiny noise, ½ per unit-variance direction at t = 1, and the
    total variance over t, tr C / t, at huge noise.
    """
    from multigeom.geometry import compute_gaussian_env

    # two directions of variance 1 and one of variance 0
    variances = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
    log_t = torch.log(torch.tensor([1e-9, 1.0, 1e9], dtype=torch.float64))
    curve = compute_gaussian_env(variances, log_t)

    # tiny noise: the two directions with variance (a direction without any is known);
    # t = 1: 1/2 each; huge noise: the total variance over t
    expected = torch.tensor([2.0, 1.0, 2e-9], dtype=torch.float64)
    torch.testing.assert_close(curve, expected, rtol=1e-6, atol=0)


def test_compute_information_on_known_curves():
    """
    Verify that `compute_information` (`multigeom.analysis`) gives half the area under
    the curve of each image (trapezoids over log t), the mean over the images and its
    standard error, on two piecewise-linear curves of known area.
    """
    from multigeom.analysis import compute_information

    # two images whose curves are 1 and 3 between log t = 0 and 2, then fall linearly
    # to 0 at log t = 4: their areas are 3 and 9
    log_t = torch.tensor([0.0, 2.0, 4.0], dtype=torch.float64)
    curves = torch.tensor([[1.0, 1.0, 0.0], [3.0, 3.0, 0.0]], dtype=torch.float64)
    mean, sem = compute_information(curves, log_t)

    # half of each area (1.5 and 4.5 nats), their mean, and its standard error (the
    # standard deviation of the two, 3 / √2, over √2)
    assert abs(mean - 3.0) < 1e-12
    assert abs(sem - 1.5) < 1e-12


def test_compute_normalized_on_known_values():
    """
    Verify that `compute_normalized` (`multigeom.analysis`) divides the values by the
    largest value of the reference: a curve by its own maximum, and eigenvalues by the
    largest leading eigenvalue, so that the second ones keep their size relative to it.
    """
    import numpy as np

    from multigeom.analysis import compute_normalized

    # a curve over its own maximum: its peak becomes 1, a negative value stays negative
    curve = np.array([-0.5, 1.0, 4.0, 2.0])
    np.testing.assert_allclose(compute_normalized(curve, curve), [-0.125, 0.25, 1, 0.5])

    # eigenvalues [levels, k] over the largest leading eigenvalue (first column)
    values = np.array([[2.0, 1.0], [8.0, 4.0], [4.0, 0.5]])
    expected = np.array([[0.25, 0.125], [1.0, 0.5], [0.5, 0.0625]])
    np.testing.assert_allclose(compute_normalized(values, values[:, 0]), expected)


def test_compute_bootstrap_resamples_the_same_images_for_every_area():
    """
    Verify that `compute_bootstrap` (`multigeom.analysis`) draws the same images for
    every area in each resample (two areas with the same curves get the same resampled
    means), that curves equal for every image give a band of zero width, and that
    `compute_band` gives the 2.5th and 97.5th percentiles of the resamples.
    """
    import numpy as np

    from multigeom.analysis import compute_band, compute_bootstrap

    # random curves of 50 images at 4 points, given to two areas, and a constant one
    rng = np.random.default_rng(0)
    curves = rng.normal(size=(50, 4))
    constant = np.tile([1.0, 2.0, 4.0, 2.0], (50, 1))
    out = compute_bootstrap({"a": curves, "b": curves, "c": constant}, 300, seed=1)

    # paired: the same resamples for areas a and b
    np.testing.assert_array_equal(out["a"]["mean"], out["b"]["mean"])
    assert out["a"]["mean"].shape == (300, 4)

    # a curve that does not vary across the images: every resample is that curve
    band = compute_band(out["c"]["normalized"])
    np.testing.assert_allclose(band, [[0.25, 0.5, 1, 0.5]] * 2)

    # the band of the resamples, against numpy's percentiles
    expected = np.percentile(out["a"]["mean"], [2.5, 97.5], axis=0)
    np.testing.assert_allclose(compute_band(out["a"]["mean"]), expected)


def test_compute_fisher_per_cell_on_a_linear_encoder():
    """
    Verify that `compute_fisher_per_cell` (`multigeom.geometry`) gives, for an encoder
    whose responses are a known linear map A of the image, each neuron's pattern A_n / σ
    and Fisher information ‖A_n‖² / σ², and that their total J(x) has the eigenvalues of
    AᵀA / σ², whose sum is that of the neurons'.
    """
    import numpy as np

    from multigeom.geometry import compute_fisher_per_cell

    # a linear encoder of 3 neurons on 8 x 8 images, and the response noise
    weight = torch.randn(3, 64, generator=torch.Generator().manual_seed(0))
    encoder = torch.nn.Sequential(
        torch.nn.Flatten(), torch.nn.Linear(64, 3, bias=False)
    )
    encoder[1].weight.data = weight
    sigma = 0.5
    out = compute_fisher_per_cell(
        torch.zeros(1, 1, 8, 8), encoder, sigma, torch.device("cpu")
    )

    # each neuron: its pattern and its Fisher information
    rows = weight.double().numpy() / sigma
    np.testing.assert_allclose(out["cells"], rows, rtol=1e-6)
    np.testing.assert_allclose(out["cell_values"], (rows**2).sum(axis=1), rtol=1e-6)

    # their total: the eigenvalues of AᵀA / σ², which add up to the neurons' values
    expected = np.sort(np.linalg.eigvalsh(rows @ rows.T))[::-1]
    np.testing.assert_allclose(out["values"], expected, rtol=1e-6)
    np.testing.assert_allclose(out["values"].sum(), out["cell_values"].sum())
