"""
Fitting the readout of an encoder to neural responses.

The backbone of an encoder is kept fixed, so its features are computed once and the readout is fitted on the cached
features (fast, and the same for a pretrained backbone as for one that you trained yourself). `fit_readout_on_features`
is the loop and the only place where a readout is fitted: `NeuralEncoder.fit_readout` caches the features and calls it,
and so do the paper's scripts (`multigeom.fit.train_encoder`, `multigeom.fit.layer_sweep`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


def correlation_per_neuron(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """Pearson correlation per neuron over all images. Inputs: [N, n_neurons]. A constant column gives 0."""
    corrs = []
    for n in range(pred.shape[1]):
        p = pred[:, n].detach().cpu().numpy()
        t = target[:, n].detach().cpu().numpy()
        if p.std() > 1e-8 and t.std() > 1e-8:
            corrs.append(float(np.corrcoef(p, t)[0, 1]))
        else:
            corrs.append(0.0)
    return np.array(corrs)


@dataclass
class FitReport:
    """What `fit_readout` did: the loss and correlation per epoch, and the epoch whose readout was kept."""

    history: dict = field(default_factory=lambda: {"epoch": [], "train_loss": [], "val_loss": [], "val_corr": []})
    best_epoch: int = 0
    best_val_corr: float = float("nan")
    n_train: int = 0
    n_val: int = 0

    def __str__(self) -> str:
        return (f"fitted on {self.n_train} images, validated on {self.n_val}: best epoch {self.best_epoch}, "
                f"validation correlation {self.best_val_corr:.3f}")


def fit_readout_on_features(
    readout: torch.nn.Module,
    features: torch.Tensor,
    responses: torch.Tensor,
    *,
    epochs: int = 100,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    val_fraction: float = 0.05,
    seed: int = 42,
    grad_clip: float = 1.0,
    device: str | torch.device | None = None,
    log: Callable[[str], None] | None = print,
) -> FitReport:
    """
    Fit `readout` (features [N, C, S, S] -> responses [N, n_neurons]) with AdamW, a cosine schedule and mean squared
    error. A fixed-seed image-level split keeps `val_fraction` of the images for validation; the readout of the epoch
    with the highest mean validation correlation is restored at the end. `features` and `responses` are used on
    their own device (put them on the GPU for speed), unless `device` is given: each batch is then moved there, so the
    cached features can stay on the CPU while the readout is on the GPU.
    """
    n = len(features)
    if len(responses) != n:
        raise ValueError(f"{n} feature maps but {len(responses)} response vectors")
    n_val = int(round(val_fraction * n))
    n_tr = n - n_val
    if n_tr < 1:
        raise ValueError("no image is left for training; lower val_fraction")
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    tr_idx, val_idx = perm[:n_tr], perm[n_tr:]
    if log:
        log(f"  train/val split: {n_tr}/{n_val}")

    optimizer = torch.optim.AdamW(readout.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    report = FitReport(n_train=n_tr, n_val=n_val)
    best_val_corr, best_state = -float("inf"), None

    for epoch in range(1, epochs + 1):
        readout.train()
        perm_tr = tr_idx[torch.randperm(len(tr_idx))]
        train_loss_sum, n_batches = 0.0, 0
        for i in range(0, len(perm_tr), batch_size):
            idx = perm_tr[i: i + batch_size]
            x, y = features[idx], responses[idx]
            if device is not None:
                x, y = x.to(device), y.to(device)
            pred = readout(x)
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(readout.parameters(), grad_clip)
            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1
        train_loss = train_loss_sum / n_batches

        readout.eval()
        val_loss, val_corr = float("nan"), float("nan")
        if n_val:
            with torch.no_grad():
                x, y = features[val_idx], responses[val_idx]
                if device is not None:
                    x, y = x.to(device), y.to(device)
                val_pred = readout(x)
                val_loss = F.mse_loss(val_pred, y).item()
                val_corr = correlation_per_neuron(val_pred, y).mean()
        scheduler.step()
        for key, value in (("epoch", epoch), ("train_loss", train_loss), ("val_loss", val_loss),
                           ("val_corr", float(val_corr))):
            report.history[key].append(value)

        score = val_corr if n_val else -train_loss           # without validation images, keep the lowest training loss
        if score > best_val_corr:
            best_val_corr = score
            report.best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in readout.state_dict().items()}
        if log and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            log(f"  epoch {epoch:4d}/{epochs}: train={train_loss:.4f}  val={val_loss:.4f}  "
                f"corr={val_corr:.3f}  (best={best_val_corr:.3f})")

    readout.load_state_dict(best_state)
    readout.eval()
    report.best_val_corr = float(best_val_corr) if n_val else float("nan")
    return report
