"""
The encoder contract: what the pipeline requires of an encoder f: image -> responses, and a check that enforces it.

A user can bring any fitted PyTorch model, as long as it satisfies these rules. `validate_encoder` runs a
few forward and backward passes on random images and raises `EncoderContractError` (naming the rule and what
to change) at the first violation; softer problems are returned as warnings.

    R1  an `nn.Module`
    R2  input [B, 1, H, W] float with values in [-1, 1] (the model does any rescaling or channel repeat itself)
    R3  output [B, N] float, finite, the same N for every call and batch size (and `n_neurons`, if present, == N)
    R4  deterministic in eval mode (no dropout or noise at inference)
    R5  differentiable w.r.t. the input through `forward` (no `torch.no_grad()`, no `detach()`): needed for J(x)
    R6  samples independent within a batch (no batch-statistics layers in train mode)
    R7  does not modify its input in place
    R8  not constant across images (warning)
    R9  response scale comparable to the noise sigma, i.e. responses of order 1 (warning)

Which step needs what: training the diffusion model on the encoder's responses needs R1-R4, R6 and R9;
estimating G(x) and J(x) needs all of them (R5 for J(x)). Not checkable here: the quality of the fit, how
the responses were measured, and that the images are 64x64 grayscale natural images.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn

@runtime_checkable
class Encoder(Protocol):
    """
    What the pipeline asks of an encoder f: image -> responses (see the rules above). Any `torch.nn.Module` that maps an
    image batch [B, 1, 64, 64] in [-1, 1] to responses [B, N] satisfies it; `n_neurons` is optional (the width of the
    output is used otherwise). `validate_encoder` checks the behaviour that a Protocol cannot express.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


TRAINING_RULES = ("R1", "R2", "R3", "R4", "R6", "R9")
ALL_RULES = ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9")


class EncoderContractError(ValueError):
    """The encoder breaks a rule of the contract; `rule` is its id (R1-R9)."""

    def __init__(self, rule: str, message: str):
        super().__init__(f"encoder contract, rule {rule}: {message}")
        self.rule = rule


@dataclass
class EncoderReport:
    n_neurons: int
    response_std: float                          # mean over neurons of the std across the random test images
    warnings: list[str] = field(default_factory=list)


def _forward(encoder, x, rule: str = "R2"):
    try:
        return encoder(x)
    except Exception as e:                       # report what the model itself complained about
        raise EncoderContractError(rule, f"forward failed on an input of shape {tuple(x.shape)} with values in "
                                         f"[{float(x.min()):.0f}, {float(x.max()):.0f}]: {type(e).__name__}: {e}") from e


def validate_encoder(encoder, *, device=None, image_shape=(1, 64, 64), sigma: float = 0.22,
                     purpose: str = "all", n_images: int = 8, seed: int = 0, atol: float = 1e-5) -> EncoderReport:
    """
    Check `encoder` against the contract. `purpose` is "training" (rules for training the diffusion model),
    "inference" or "all" (both). Raises EncoderContractError on a violation; warnings are in the report.
    The check puts the encoder in eval mode and leaves it there.
    """
    if purpose not in ("training", "inference", "all"):
        raise ValueError("purpose must be 'training', 'inference' or 'all'")
    rules = TRAINING_RULES if purpose == "training" else ALL_RULES
    warnings: list[str] = []

    if not isinstance(encoder, nn.Module):                                                       # R1
        raise EncoderContractError("R1", f"the encoder must be a torch.nn.Module, got {type(encoder).__name__}")
    encoder.eval()
    if device is None:
        device = next(encoder.parameters()).device if any(True for _ in encoder.parameters()) else torch.device("cpu")
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.rand(n_images, *image_shape, generator=g) * 2 - 1).to(device)

    with torch.no_grad():
        for value in (-1.0, 1.0):                                                                # R2: the extremes
            _forward(encoder, torch.full((2, *image_shape), value, device=device))
        before = x.clone()
        out = _forward(encoder, x)
        if "R7" in rules and not torch.equal(before, x):
            raise EncoderContractError("R7", "forward modified its input in place; work on a copy")

        # R3
        if not isinstance(out, torch.Tensor) or out.ndim != 2 or out.shape[0] != n_images:
            shape = tuple(out.shape) if isinstance(out, torch.Tensor) else type(out).__name__
            raise EncoderContractError("R3", f"output must be a tensor of shape [B, N]; got {shape} for B={n_images}")
        if not out.is_floating_point() or not torch.isfinite(out).all():
            raise EncoderContractError("R3", "output must be floating point and finite (found NaN, inf or a non-float dtype)")
        n = out.shape[1]
        if n < 1:
            raise EncoderContractError("R3", "output has no neurons (N = 0)")
        out_b4 = _forward(encoder, x[:4])
        if out_b4.shape != (4, n):
            raise EncoderContractError("R3", f"N changes with the batch size: {n} for B={n_images}, "
                                             f"{tuple(out_b4.shape)} for B=4")
        declared = getattr(encoder, "n_neurons", None)
        if declared is not None and int(declared) != n:
            raise EncoderContractError("R3", f"encoder.n_neurons = {declared} but the output has {n} columns")

        # R4: eval-mode determinism
        if not torch.allclose(out, _forward(encoder, x), atol=atol):
            raise EncoderContractError("R4", "two forward passes on the same input differ in eval mode "
                                             "(dropout or random noise left on at inference?)")

        # R6: independence within a batch
        single = _forward(encoder, x[:1])
        # GPU kernels differ slightly with the batch size (measured ~1e-3 on the reference encoders), so the
        # tolerance is 1% of the response scale; batch coupling changes the responses by order one.
        tol = 1e-2 * max(float(out.abs().max()), 1e-3)
        if not torch.allclose(single, out[:1], atol=tol, rtol=0):
            raise EncoderContractError("R6", "the response to an image changes with the other images in the batch "
                                             "(batch-statistics layer in train mode, or a cross-sample operation)")

    # R5: gradient w.r.t. the input through forward
    if "R5" in rules:
        xg = x[:1].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            y = _forward(encoder, xg, "R5")
            if not y.requires_grad:
                raise EncoderContractError("R5", "the output does not depend differentiably on the input "
                                                 "(torch.no_grad(), detach() or a non-differentiable operation in forward)")
            y.sum().backward()
        if xg.grad is None or not torch.isfinite(xg.grad).all():
            raise EncoderContractError("R5", "the gradient w.r.t. the input is missing or not finite")
        if float(xg.grad.abs().max()) == 0.0:
            raise EncoderContractError("R5", "the gradient w.r.t. the input is identically zero (J(x) would be 0)")

    # R8, R9: warnings
    std = out.std(dim=0)
    if float(std.max()) < 1e-8:
        warnings.append("R8: the responses are identical for different images: the geometry will be degenerate")
    mean_std = float(std.mean())
    if "R9" in rules and not (0.1 <= mean_std <= 10.0):
        warnings.append(f"R9: responses have std {mean_std:.3g} over random test images, while the noise sigma is "
                        f"{sigma}; the pipeline expects responses of order 1 (for example z-scored). "
                        "Rescale the model's output or set sigma accordingly. (Random noise images only "
                        "approximate natural-image statistics.)")
    return EncoderReport(n_neurons=n, response_std=mean_std, warnings=warnings)


def ensure_valid(encoder, **kwargs) -> None:
    """validate_encoder once per encoder object and settings; warnings are issued with `warnings.warn`."""
    import warnings as _warnings
    key = tuple(sorted((k, str(v)) for k, v in kwargs.items()))
    done = getattr(encoder, "_contract_validated", set())
    if key in done:
        return
    report = validate_encoder(encoder, **kwargs)
    for w in report.warnings:
        _warnings.warn(w, stacklevel=3)
    done.add(key)
    try:
        encoder._contract_validated = done
    except AttributeError:
        pass
