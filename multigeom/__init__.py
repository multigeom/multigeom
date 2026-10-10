"""
multigeom: the multi-scale Fisher information geometry of neural population codes, estimated with conditional diffusion
models (Azeglio, Laquitaine, Ferrari, Chalk; NeurIPS 2026).

Bring an encoder that is already fitted to your neural data (any PyTorch model that follows the contract of
`multigeom.encoder`) and compute

    compute_fisher(image, encoder, sigma)         the Fisher information J(x): its top eigenvalues and eigenvectors
    estimate_g(image, encoder, unet, ...)         the multi-scale Fisher metric G(x) (needs a trained diffusion model)
    align(eigvecs_a, eigvecs_b)                   the subspace alignment of two sets of eigenvectors
    bures_wasserstein(...)                        the Bures-Wasserstein distance between two reconstructed metrics
    validate_encoder(encoder)                     check that an encoder follows the contract

The core needs torch, numpy and scipy (and `diffusers` for the diffusion model). Reference encoders (extra `[vision]`),
fitting an encoder to neural data (extra `[fit]`), figures (extra `[viz]`) and the Modal launcher (extra `[modal]`) are
optional. Subpackages: `geometry`, `diffusion`, `pipeline`, `data`, `encoders`, `fit`, `viz`.
"""
from .encoder import Encoder, EncoderContractError, EncoderReport, validate_encoder
from .geometry.estimator import EstimatorOutput
from .geometry.estimator import conditional_pair_estimate as estimate_g
from .geometry.fisher import FisherOutput
from .geometry.fisher import fisher_eigendecomposition as compute_fisher
from .geometry.metrics import bures_wasserstein
from .geometry.metrics import subspace_alignment as align

__version__ = "1.0.0"

__all__ = [
    "Encoder", "EncoderContractError", "EncoderReport", "validate_encoder",
    "compute_fisher", "estimate_g", "align", "bures_wasserstein",
    "FisherOutput", "EstimatorOutput", "__version__",
]
