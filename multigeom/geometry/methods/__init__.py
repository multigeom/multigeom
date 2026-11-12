"""
Methods that estimate G(x): the part of the estimator that differs between them.

All methods share the skeleton of `multigeom.geometry.estimator.estimate_g`: the log-τ
grid and its trapezoid weights, the Monte-Carlo loop, the matrix D and its SVD. A method
supplies only the rows of D: for noisy images x_t drawn at the DDPM step t_j, it returns
a [B, P] array of differences in noise-prediction units, whose covariance integrates
to G(x) once weighted. A method is a class with this interface:

    class MyMethod:
        # rows are weighted by sqrt(w_j / (row_normalization · N_MC))
        row_normalization = 1.0

        def __init__(self, *, image, encoder, model, alphas_cumprod, cfg, device,
                     generator=None): ...

        def sample_rows(self, x_t, t, t_j) -> Tensor:    # [B, P]

        # optional: reported in the output (EstimatorOutput.diagnostics)
        def diagnostics(self) -> dict: ...

`estimate_g` is decorated with `torch.no_grad`: a method that needs gradients (for
instance of the encoder) enables them itself with `torch.enable_grad()`. All random
draws must use `generator`, so that an image is reproducible on its own.

    register_method("my_method", MyMethod)
    estimate_g(image, encoder, model, alphas_cumprod, cfg, device, method="my_method")

The method is chosen by the `method` argument of `estimate_g`, or else by the
`estimator` entry of the estimator config (default: "conditional_pair").
"""

from __future__ import annotations

from .conditional_pair import ConditionalPair

DEFAULT_METHOD = "conditional_pair"

_METHODS: dict[str, type] = {DEFAULT_METHOD: ConditionalPair}


def register_method(name: str, method: type, overwrite: bool = False) -> None:
    """
    Register a method of G(x) under `name`. A name that is already taken raises a
    ValueError unless `overwrite` is set.
    """
    if name in _METHODS and not overwrite:
        raise ValueError(
            f"the method {name!r} is already registered; "
            "pass overwrite=True to replace it"
        )
    _METHODS[name] = method


def available_methods() -> list[str]:
    """The names of the registered methods."""
    return sorted(_METHODS)


def get_method(name: str) -> type:
    """The class registered under `name`; an unknown name raises a ValueError."""
    if name not in _METHODS:
        raise ValueError(
            f"unknown method {name!r}; the registered methods are {available_methods()}"
        )
    return _METHODS[name]


__all__ = [
    "ConditionalPair",
    "DEFAULT_METHOD",
    "available_methods",
    "get_method",
    "register_method",
]
