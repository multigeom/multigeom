"""
`NeuralEncoder`: an image-to-neural-response model made of a backbone and a readout, with the methods to use it:

    encoder = NeuralEncoder(backbone, n_neurons=50, readout="gaussian")   # build
    encoder.fit_readout(images, responses)                                 # fit the readout to neural data
    responses = encoder.predict(images)                                    # predict neural responses
    encoder.save("my_encoder.pt");  encoder = NeuralEncoder.load("my_encoder.pt")

    encoder = load_encoder("inception", area="V1")                         # an encoder fitted to macaque V1

The input is the canonical image batch [B, 1, 64, 64] in [-1, 1]; the output is [B, n_neurons]. A `NeuralEncoder`
follows the encoder contract of `multigeom.encoders.contract`, so it plugs into `compute_fisher`, `compute_g` and the pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .fitting import FitReport, fit_readout_on_features
from . import backbones, readout  # noqa: F401  (registers the built-in backbones and readouts, whatever the import order)
from .registry import BACKBONES, READOUTS, describe

CHECKPOINT_FORMAT = 1
DEFAULT_REPO = "steevelaquitaine/multigeom"         # Hugging Face repository with the checkpoints of the paper


def _package_version() -> str:
    from .. import __version__
    return __version__


class NeuralEncoder(nn.Module):
    """
    A backbone followed by a readout.

    Args:
        backbone: an `nn.Module` mapping images [B, 1, 64, 64] to features [B, C, S, S], with attributes
            `out_channels` and `out_spatial`. Pretrained backbones can be frozen already; a backbone that you trained
            is frozen by `fit_readout`.
        n_neurons: number of neurons to predict.
        readout: the name of a registered readout ("gaussian", "linear", "factorized", or one you registered) or a
            readout module (`features [B, C, S, S] -> [B, n_neurons]`).
        dropout: dropout on the feature map while the readout is fitted (used when `readout` is a name).
    """

    def __init__(self, backbone: nn.Module, n_neurons: int, readout: str | nn.Module = "gaussian",
                 dropout: float = 0.25):
        super().__init__()
        self.backbone = backbone
        self.n_neurons = int(n_neurons)
        if isinstance(readout, str):
            if readout not in READOUTS:
                raise ValueError(f"unknown readout {readout!r}; available: {sorted(READOUTS)}")
            readout = READOUTS[readout](n_neurons=self.n_neurons, n_channels=backbone.out_channels,
                                        spatial=backbone.out_spatial, dropout=dropout)
        self.readout = readout
        self.metadata: dict = {}                      # what a checkpoint said about the fit (see `load`)
        self._backbone_frozen = not any(p.requires_grad for p in backbone.parameters()) and \
            any(True for _ in backbone.parameters())

    # ── forward ──────────────────────────────────────────────────────────
    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        """Convert the canonical image batch to what the backbone expects (nothing, here; see the reference encoders)."""
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, 64, 64] in [-1, 1]  ->  responses [B, n_neurons]."""
        # No torch.no_grad() here: the gradient with respect to the image has to flow (J(x), encoder contract rule R5).
        return self.readout(self.backbone(self._prepare(x)))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """The backbone's feature map only (no readout, no gradient): what `fit_readout` caches."""
        with torch.no_grad():
            return self.backbone(self._prepare(x))

    # ── backbone freezing ────────────────────────────────────────────────
    def freeze_backbone(self) -> "NeuralEncoder":
        """Stop training the backbone: its parameters get no gradient and its BatchNorm statistics stay fixed."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._backbone_frozen = True
        self.backbone.eval()
        return self

    def train(self, mode: bool = True):  # noqa: A003
        """Set the mode; a frozen backbone stays in evaluation mode."""
        super().train(mode)
        if self._backbone_frozen:
            self.backbone.eval()
        return self

    # ── predict ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, images, batch_size: int = 256, device: str | torch.device | None = None) -> torch.Tensor:
        """
        Predicted neural responses [N, n_neurons] (on the CPU) for `images` [N, 1, 64, 64] in [-1, 1], computed in
        batches in evaluation mode. `images` is a tensor, or a dataset or loader that yields image batches (or
        (images, labels) pairs). The encoder is moved to `device` if one is given, and the model's mode is restored.
        """
        if device is not None:
            self.to(device)
        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        try:
            out = []
            for batch in _batches(images, batch_size):
                out.append(self(batch.to(device)).cpu())
            return torch.cat(out, dim=0)
        finally:
            self.train(was_training)

    # ── fit ──────────────────────────────────────────────────────────────
    def fit_readout(self, images: torch.Tensor, responses: torch.Tensor, *, epochs: int = 100, lr: float = 3e-4,
                    weight_decay: float = 1e-4, batch_size: int = 64, val_fraction: float = 0.05, seed: int = 42,
                    device: str | torch.device | None = None, verbose: bool = True) -> FitReport:
        """
        Fit the readout to neural data and keep the best epoch.

        The backbone is frozen first, its features are computed once for `images` [N, 1, 64, 64], and the readout is
        trained on them to predict `responses` [N, n_neurons] (AdamW, cosine schedule, mean squared error, a
        validation split of `val_fraction` of the images). `seed` fixes the split and, through torch's global generator,
        the order of the batches and the dropout, so a fit can be repeated exactly. Returns a `FitReport`.
        """
        if responses.shape != (len(images), self.n_neurons):
            raise ValueError(f"responses should be [{len(images)}, {self.n_neurons}], got {tuple(responses.shape)}")
        if device is not None:
            self.to(device)
        device = next(self.parameters()).device
        torch.manual_seed(seed)
        self.freeze_backbone()
        self.eval()
        feats = torch.cat([self.features(b.to(device)) for b in _batches(images, 256)], dim=0)
        report = fit_readout_on_features(
            self.readout, feats, responses.to(device), epochs=epochs, lr=lr, weight_decay=weight_decay,
            batch_size=batch_size, val_fraction=val_fraction, seed=seed, log=print if verbose else None)
        self.eval()
        return report

    # ── architecture, save, load ─────────────────────────────────────────
    @property
    def config(self) -> dict:
        """The architecture as plain data: enough to rebuild the encoder (without its weights)."""
        return {
            "backbone": describe(self.backbone, BACKBONES, "backbone"),
            "n_neurons": self.n_neurons,
            "readout": describe(self.readout, READOUTS, "readout"),
        }

    @classmethod
    def from_config(cls, config: dict) -> "NeuralEncoder":
        """Build an encoder (with fresh weights) from `config`, as returned by `encoder.config`."""
        if "reference" in config:
            from .reference import build_reference
            return build_reference(config)
        b, r = config["backbone"], config["readout"]
        for kind, spec, registry in (("backbone", b, BACKBONES), ("readout", r, READOUTS)):
            if spec["name"] not in registry:
                raise ValueError(f"the {kind} {spec['name']!r} is not registered; import or register it first "
                                 f"(available: {sorted(registry)})")
        backbone = BACKBONES[b["name"]](**b["kwargs"])
        readout = READOUTS[r["name"]](**r["kwargs"])
        return cls(backbone, n_neurons=config["n_neurons"], readout=readout)

    def save(self, path: str | Path, **metadata: Any) -> Path:
        """
        Save the weights, the architecture and `metadata` (anything picklable, e.g. the area or the test correlation)
        to one file. Reload with `NeuralEncoder.load(path)`.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format_version": CHECKPOINT_FORMAT,
            "model_state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
            "config": {"architecture": self.config},
            "metadata": metadata,
            "multigeom_version": _package_version(),
        }, path)
        return path

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device | None = None) -> "NeuralEncoder":
        """
        Rebuild an encoder from a checkpoint written by `save` (or one of the paper's checkpoints) and load its weights.
        What the checkpoint says about the fit (the metadata given to `save`, or the paper's validation and test
        correlations) is in `encoder.metadata`.
        """
        ckpt = _read_checkpoint(path)
        encoder = _encoder_from_checkpoint(ckpt)
        encoder.load_state_dict(ckpt["model_state_dict"])
        encoder.eval()
        encoder.metadata = _metadata_of(ckpt)
        if device is not None:
            encoder.to(device)
        return encoder

    def load_checkpoint(self, path: str | Path, strict: bool = True) -> "NeuralEncoder":
        """Load the weights of a checkpoint into this encoder (its architecture has to match)."""
        ckpt = _read_checkpoint(path)
        device = next(self.parameters()).device
        self.load_state_dict({k: v.to(device) for k, v in ckpt["model_state_dict"].items()}, strict=strict)
        self.eval()
        return self

    def summary(self) -> str:
        """One line per part: the backbone, the readout and the number of trainable parameters."""
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in self.parameters())
        return (f"{type(self).__name__}: backbone {type(self.backbone).__name__} "
                f"(out [C={self.backbone.out_channels}, {self.backbone.out_spatial}, {self.backbone.out_spatial}]), "
                f"readout {type(self.readout).__name__}, {self.n_neurons} neurons, "
                f"{n_train:,} trainable / {n_all:,} parameters")


def _batches(images, batch_size: int):
    """Image batches from a tensor, a dataset, or a loader (pairs are reduced to their first element)."""
    if isinstance(images, torch.Tensor):
        for i in range(0, len(images), batch_size):
            yield images[i: i + batch_size]
        return
    if isinstance(images, torch.utils.data.DataLoader):
        loader = images
    else:
        loader = torch.utils.data.DataLoader(images, batch_size=batch_size, shuffle=False)
    for batch in loader:
        yield batch[0] if isinstance(batch, (tuple, list)) else batch


def _read_checkpoint(path: str | Path) -> dict:
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        raise ValueError(f"{path} is not an encoder checkpoint (no 'model_state_dict')")
    return ckpt


def _metadata_of(ckpt: dict) -> dict:
    """What the checkpoint says about the encoder: the metadata given to `save`, or the fit scores of the paper's checkpoints."""
    if "metadata" in ckpt:
        return dict(ckpt["metadata"])
    meta = {k: float(ckpt[k]) for k in ("val_corr", "test_corr", "test_corr_noise_corrected") if k in ckpt}
    area = ckpt.get("config", {}).get("area")
    return ({"area": area} if area else {}) | meta


def _encoder_from_checkpoint(ckpt: dict) -> NeuralEncoder:
    config = ckpt.get("config", {})
    if "architecture" in config:
        return NeuralEncoder.from_config(config["architecture"])
    if "encoder" in config:                                          # the checkpoints of the paper: {"encoder": name, "area": ...}
        from .reference import build_reference
        n_neurons = int(ckpt["model_state_dict"]["readout.mu"].shape[0])
        return build_reference({"reference": str(config["encoder"]).lower(), "n_neurons": n_neurons})
    raise ValueError("the checkpoint does not say which architecture it holds (no config); build the encoder and use "
                     "`load_checkpoint` to load its weights")


def load_encoder(name_or_path: str | Path, area: str | None = None, *, device: str | torch.device | None = None,
                 repo_id: str = DEFAULT_REPO, revision: str | None = None) -> NeuralEncoder:
    """
    Load an encoder that is already fitted.

    * `load_encoder("inception", area="V1")`: the encoder of the paper (`"inception"` or `"resnet50"`, area `"V1"` or
      `"V4"`), downloaded from the Hugging Face repository `repo_id` the first time and cached after that. The
      pretrained backbone weights come from torchvision.
    * `load_encoder("path/to/encoder.pt")`: a checkpoint on disk, saved by `NeuralEncoder.save` or by the pipeline.
    """
    path = Path(str(name_or_path))
    if path.suffix in (".pt", ".pth", ".ckpt") or path.exists():
        if not path.exists():
            raise FileNotFoundError(path)
        return NeuralEncoder.load(path, device=device)
    if area is None:
        raise ValueError(f"{name_or_path!r} is not a file: to download an encoder of the paper give its area, "
                         f"load_encoder({str(name_or_path)!r}, area='V1')")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as err:                                       # pragma: no cover
        raise ImportError("downloading checkpoints needs huggingface_hub (pip install huggingface_hub)") from err
    local = hf_hub_download(repo_id=repo_id, filename=f"encoders/{name_or_path}/{area}/best_model.pt",
                            revision=revision)
    return NeuralEncoder.load(local, device=device)
