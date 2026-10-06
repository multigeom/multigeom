"""
Closed-form Fisher information baseline J(x) used in the paper.

For the Gaussian observation model p(r | x) = N(f_φ(x), σ² I), the classical
Fisher information matrix is

    J(x) = (1 / σ²) · J_{f_φ}(x)ᵀ J_{f_φ}(x)

where J_{f_φ}(x) ∈ ℝ^{n × P} is the Jacobian of the encoder mean f_φ at x,
with n = 50 target neurons and P = 64×64 = 4096 pixels.

Because J_{f_φ} has rank at most n ≪ P, we extract the top-k eigenpairs of
J(x) directly from the SVD of J_{f_φ} = U S Vᵀ:
    eigenvalues  = S² / σ²
    eigenvectors = rows of Vᵀ
This avoids ever forming the P × P outer product (which would be 4096² = 16M
entries).

The Jacobian itself is materialised row-by-row by reverse-mode autograd:
one backward pass per neuron through (input → frozen backbone → readout).
The backbone parameters are frozen (`requires_grad = False`), so gradients
of the response w.r.t. the input flow through but no gradient is computed
for the backbone weights.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FisherOutput:
    eigenvalues: torch.Tensor   # [k]   top-k eigenvalues of J(x), descending
    eigenvectors: torch.Tensor  # [k, P] rows are top-k eigenvectors of J(x)


def fisher_eigendecomposition(
    image: torch.Tensor,
    encoder,
    sigma: float,
    top_k: int = 50,
    device: torch.device | None = None,
) -> FisherOutput:
    """
    Compute the top-k eigenpairs of J(x) = (1/σ²) J_{f_φ}ᵀ J_{f_φ}.

    Args:
        image: [1, 1, 64, 64] in [-1, 1] (canonical pipeline coords).
        encoder: f_φ — built by `src.encoders.build_encoder`. Backbone is frozen.
        sigma: σ from the Gaussian observation model.
        top_k: number of eigenpairs to return.
        device: target device. Defaults to encoder's parameter device.

    Returns:
        FisherOutput with `eigenvalues` [k] and `eigenvectors` [k, P=4096].
    """
    encoder.eval()
    if device is None:
        device = next(encoder.parameters()).device

    if image.dim() != 4 or image.shape[0] != 1:
        raise ValueError(f"Expected single-image batch [1, C, H, W], got {tuple(image.shape)}")
    image = image.to(device)

    # Forward pass with gradient tracking on the input only. The backbone's
    # parameters have requires_grad = False (frozen), but gradients of the
    # output w.r.t. the *input* flow through it just fine.
    x = image.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        # Build the graph explicitly (input conversion → backbone → readout)
        # rather than through encoder.forward, which runs the backbone under
        # torch.no_grad(). enable_grad also makes this work when the caller
        # is itself inside a no_grad block.
        x_bb = encoder._to_backbone_input(x)
        features = encoder.backbone(x_bb)
        responses = encoder.readout(features)        # [1, n_neurons]

        n_neurons = responses.shape[1]
        P = x[0].numel()
        J_f = torch.zeros(n_neurons, P, device=device)
        for i in range(n_neurons):
            if x.grad is not None:
                x.grad.zero_()
            responses[0, i].backward(retain_graph=(i < n_neurons - 1))
            J_f[i] = x.grad.detach().reshape(-1).clone()

    # SVD of the n × P Jacobian (n ≪ P → "thin" SVD).
    # If J_f = U S Vᵀ, then J_fᵀ J_f = V S² Vᵀ.
    U, S, Vh = torch.linalg.svd(J_f, full_matrices=False)
    eigenvalues = (S ** 2) / (sigma ** 2)
    eigenvectors = Vh                                # [n_neurons, P]

    k = min(top_k, n_neurons)
    return FisherOutput(
        eigenvalues=eigenvalues[:k].detach().cpu().clone(),
        eigenvectors=eigenvectors[:k].detach().cpu().clone(),
    )
