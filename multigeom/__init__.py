"""
multigeom: the multi-scale Fisher information geometry of neural population codes, estimated with conditional diffusion
models (Azeglio, Laquitaine, Ferrari, Chalk; NeurIPS 2026).

Each group of functions is a subpackage (and the main functions are also available directly as `multigeom.<name>`).

`multigeom.encoders`: use one that is already fitted, build your own, or bring any PyTorch model that follows the contract of
`multigeom.encoder`:

    load_encoder(name, area)                      an encoder of the paper, fitted to macaque V1 or V4 (downloaded)
    NeuralEncoder(backbone, n_neurons)            build your own encoder: fit_readout, predict, save, load
    validate_encoder(encoder)                     check that an encoder follows the contract

`multigeom.geometry`: the Fisher metric and the multi-scale metric (the latter needs the diffusion model of `multigeom.diffusion`):

    compute_fisher(image, encoder, sigma)         the Fisher information J(x): its top eigenvalues and eigenvectors
    multigeom.diffusion                           the conditional diffusion model of an encoder: train it on its responses
    compute_g(image, encoder, unet, ...)         the multi-scale Fisher metric G(x) (needs the trained diffusion model)

`multigeom.analysis`: information-geometric analyses of the metrics:

    subspace_alignment(eigvecs_a, eigvecs_b)      the subspace alignment of two sets of eigenvectors
    bures_wasserstein(...)                        the Bures-Wasserstein distance between two reconstructed metrics

Install the core with `pip install multigeom`; the extra `[paper]` adds what reproducing the paper needs (the ON-OFF
encoder and the TVSD loaders), and `[all]` also Jupyter and the Modal launcher. Subpackages: `encoders`, `geometry`,
`analysis`, `diffusion`, `pipeline`, `data`, `fit`, `viz`.
"""
from .encoder import Encoder, EncoderContractError, EncoderReport, validate_encoder
from .analysis import bures_wasserstein, subspace_alignment
from .geometry import EstimatorOutput, FisherOutput, compute_fisher, compute_g

__version__ = "1.0.0"

__all__ = [
    "Encoder", "EncoderContractError", "EncoderReport", "validate_encoder",
    "NeuralEncoder", "load_encoder",
    "compute_fisher", "compute_g", "subspace_alignment", "bures_wasserstein",
    "FisherOutput", "EstimatorOutput", "__version__",
]


def __getattr__(name):                       # the encoders need torchvision: import them only when they are asked for
    if name in ("NeuralEncoder", "load_encoder"):
        from . import encoders
        return getattr(encoders, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
