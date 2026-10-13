"""
multigeom: the multi-scale Fisher information geometry of neural population codes, estimated with conditional diffusion
models (Azeglio, Laquitaine, Ferrari, Chalk; NeurIPS 2026).

Encoders: use one that is already fitted, build your own, or bring any PyTorch model that follows the contract of
`multigeom.encoder`:

    load_encoder(name, area)                      an encoder of the paper, fitted to macaque V1 or V4 (downloaded)
    NeuralEncoder(backbone, n_neurons)            build your own encoder: fit_readout, predict, save, load
    validate_encoder(encoder)                     check that an encoder follows the contract

The Fisher metric, the diffusion model and the multi-scale metric:

    compute_fisher(image, encoder, sigma)         the Fisher information J(x): its top eigenvalues and eigenvectors
    multigeom.diffusion                           the conditional diffusion model of an encoder: train it on its responses
    compute_g(image, encoder, unet, ...)         the multi-scale Fisher metric G(x) (needs the trained diffusion model)

Information-geometric analyses of the metrics:

    align(eigvecs_a, eigvecs_b)                   the subspace alignment of two sets of eigenvectors
    bures_wasserstein(...)                        the Bures-Wasserstein distance between two reconstructed metrics

Install the core with `pip install multigeom`; the extra `[paper]` adds what reproducing the paper needs (the ON-OFF
encoder and the TVSD loaders), and `[all]` also Jupyter and the Modal launcher. Subpackages: `encoders`, `geometry`,
`diffusion`, `pipeline`, `data`, `fit`, `viz`.
"""
from .encoder import Encoder, EncoderContractError, EncoderReport, validate_encoder
from .geometry.estimator import EstimatorOutput
from .geometry.estimator import conditional_pair_estimate as compute_g
from .geometry.fisher import FisherOutput
from .geometry.fisher import fisher_eigendecomposition as compute_fisher
from .geometry.metrics import bures_wasserstein
from .geometry.metrics import subspace_alignment as align

__version__ = "1.0.0"

__all__ = [
    "Encoder", "EncoderContractError", "EncoderReport", "validate_encoder",
    "NeuralEncoder", "load_encoder",
    "compute_fisher", "compute_g", "align", "bures_wasserstein",
    "FisherOutput", "EstimatorOutput", "__version__",
]


def __getattr__(name):                       # the encoders need torchvision: import them only when they are asked for
    if name in ("NeuralEncoder", "load_encoder"):
        from . import encoders
        return getattr(encoders, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
