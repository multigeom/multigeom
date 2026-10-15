"""
Tests for the synthetic neural data, the TinyImageNet labels and the callable diffusion
functions.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from multigeom import validate_encoder
from multigeom.datasets import (
    GaborTeacher,
    TinyImageNetDataset,
    make_synthetic_neural_data,
    noise_ceiling,
)


def images(n=64, seed=0):
    return (
        torch.rand(n, 1, 64, 64, generator=torch.Generator().manual_seed(seed)) * 2 - 1
    )


# ── synthetic neural data ───────────────────────────────────────────────────
def test_teacher_follows_the_encoder_contract_with_responses_of_order_one():
    x = images(100)
    teacher = GaborTeacher(10, seed=0).calibrate(x)
    report = validate_encoder(teacher, sigma=0.22)
    assert report.n_neurons == 10 and not report.warnings
    r = teacher(x)
    torch.testing.assert_close(r.std(0), torch.ones(10), atol=1e-4, rtol=0)


def test_teacher_is_determined_by_its_seed_and_different_across_seeds():
    x = images(8)
    a, b, c = GaborTeacher(5, seed=1), GaborTeacher(5, seed=1), GaborTeacher(5, seed=2)
    torch.testing.assert_close(a(x), b(x))
    assert not torch.allclose(a(x), c(x))


def test_teacher_has_a_gradient_with_respect_to_the_image():
    x = images(2).requires_grad_(True)
    GaborTeacher(4, seed=0)(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_teacher_neuron_prefers_its_own_orientation():
    """
    A grating at the neuron's orientation and frequency drives it more than the
    orthogonal one.
    """
    teacher = GaborTeacher(1, seed=3, complex_fraction=1.0)
    p = teacher.params
    ax = np.linspace(-1, 1, 64)
    xx, yy = np.meshgrid(ax, ax)

    def grating(theta):
        u = (xx - p["centre"][0, 0]) * np.cos(theta) + (
            yy - p["centre"][0, 1]
        ) * np.sin(theta)
        return torch.tensor(
            np.cos(2 * np.pi * p["cycles"][0] * u / 2), dtype=torch.float32
        ).view(1, 1, 64, 64)

    best, orthogonal = (
        teacher(grating(p["theta"][0])).item(),
        teacher(grating(p["theta"][0] + np.pi / 2)).item(),
    )
    assert best > 3 * orthogonal


def test_synthetic_data_noise_matches_sigma_and_the_ceiling_formula():
    x = images(2000)
    teacher = GaborTeacher(6, seed=0).calibrate(x)
    noisy, clean = make_synthetic_neural_data(teacher, x, noise_sigma=0.5, seed=4)
    assert noisy.shape == clean.shape == (2000, 6)
    assert (noisy - clean).std().item() == pytest.approx(0.5, abs=0.03)
    ceiling = noise_ceiling(clean, 0.5)
    measured = np.array([np.corrcoef(noisy[:, n], clean[:, n])[0, 1] for n in range(6)])
    np.testing.assert_allclose(ceiling, measured, atol=0.03)
    again, _ = make_synthetic_neural_data(teacher, x, noise_sigma=0.5, seed=4)
    torch.testing.assert_close(again, noisy)


# ── TinyImageNet labels and subsets ─────────────────────────────────────────
@pytest.fixture
def fake_tinyimagenet(tmp_path):
    root = tmp_path / "tiny-imagenet-200"
    wnids = ["n001", "n002", "n003"]
    (root).mkdir()
    (root / "wnids.txt").write_text("\n".join(wnids) + "\n")
    for w in wnids:
        d = root / "train" / w / "images"
        d.mkdir(parents=True)
        for i in range(4):
            Image.fromarray(np.full((64, 64, 3), 40 * i, dtype=np.uint8)).save(
                d / f"{w}_{i}.JPEG"
            )
    vd = root / "val" / "images"
    vd.mkdir(parents=True)
    lines = []
    for i, w in enumerate(["n003", "n001", "n002"]):
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(
            vd / f"val_{i}.JPEG"
        )
        lines.append(f"val_{i}.JPEG\t{w}\t0\t0\t63\t63")
    (root / "val" / "val_annotations.txt").write_text("\n".join(lines) + "\n")
    return root


def test_labels_follow_the_class_order_and_subsets_keep_the_first_images(
    fake_tinyimagenet,
):
    ds = TinyImageNetDataset("train", fake_tinyimagenet, return_labels=True)
    assert len(ds) == 12 and ds.labels == [0] * 4 + [1] * 4 + [2] * 4
    x, y = ds[5]
    assert x.shape == (1, 64, 64) and y == 1
    small = TinyImageNetDataset(
        "train", fake_tinyimagenet, images_per_class=2, return_labels=True
    )
    assert len(small) == 6 and small.labels == [0, 0, 1, 1, 2, 2]
    assert TinyImageNetDataset("train", fake_tinyimagenet)[0].shape == (
        1,
        64,
        64,
    )  # images only by default


def test_validation_labels_come_from_the_annotations(fake_tinyimagenet):
    ds = TinyImageNetDataset("val", fake_tinyimagenet, return_labels=True)
    assert ds.labels == [2, 0, 1]
    with pytest.raises(ValueError, match="train split"):
        TinyImageNetDataset("val", fake_tinyimagenet, images_per_class=1)


# ── diffusion functions ─────────────────────────────────────────────────────
def test_precompute_responses_handles_images_and_pairs():
    from multigeom.diffusion.responses import precompute_responses
    from multigeom.encoders import NeuralEncoder, SmallCNNBackbone

    enc = NeuralEncoder(SmallCNNBackbone((4, 4, 4)), 5)
    x = images(10)

    class Images(torch.utils.data.Dataset):
        def __len__(self):
            return 10

        def __getitem__(self, i):
            return x[i]

    class Pairs(Images):
        def __getitem__(self, i):
            return x[i], 0

    a, b = (
        precompute_responses(enc, Images(), batch_size=4),
        precompute_responses(enc, Pairs(), batch_size=3),
    )
    assert a.shape == (10, 5)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(a, enc.predict(x))


def test_train_diffusion_runs_on_a_tiny_model_and_resumes(tmp_path):
    pytest.importorskip("diffusers")
    from multigeom.config import load_config
    from multigeom.diffusion import DiffusionConfig
    from multigeom.diffusion.model import build_diffusion_model
    from multigeom.diffusion.train import train_diffusion
    from multigeom.encoders import NeuralEncoder, SmallCNNBackbone

    enc = NeuralEncoder(SmallCNNBackbone((4, 4, 4)), 5)
    cfg = DiffusionConfig.for_encoder(enc)
    cfg.unet.block_channels = [32, 32, 32, 32]
    cfg.unet.layers_per_block = 1
    cfg.unet.cross_attention_dim = 8
    cfg.conditioning.mlp_layers = [5, 8, 8, 8]
    cfg.train.epochs, cfg.train.batch_size, cfg.train.warmup_steps = 2, 4, 1

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return 8

        def __getitem__(self, i):
            return images(8)[i]

    resp = enc.predict(images(8))
    history = train_diffusion(
        cfg,
        DS(),
        resp,
        tmp_path,
        device="cpu",
        num_workers=0,
        log=None,
        save_every=1,
        metadata={"area": "V1"},
    )
    assert history["epoch"] == [1, 2] and len(history["loss"]) == 2
    ckpt = torch.load(tmp_path / "final_model.pt", weights_only=False)
    assert (
        ckpt["epoch"] == 2
        and ckpt["area"] == "V1"
        and (tmp_path / "model_epoch_001.pt").exists()
    )
    model = build_diffusion_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    assert not (tmp_path / "last.pt").exists()  # removed when the run completes
    with pytest.raises(ValueError, match="for_encoder"):
        train_diffusion(
            load_config("diffusion"),
            DS(),
            resp,
            tmp_path / "bad",
            device="cpu",
            num_workers=0,
            log=None,
        )


# ── the whole synthetic experiment in one call ──────────────────────────────
def test_make_synthetic_experiment_splits_the_images_like_the_manual_recipe(tmp_path):
    from multigeom.datasets import make_synthetic_experiment

    root = tmp_path / "tiny-imagenet-200"
    (root / "train").mkdir(parents=True)
    vd = root / "val" / "images"
    vd.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(30):
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(
            vd / f"val_{i:02d}.JPEG"
        )

    exp = make_synthetic_experiment(
        n_neurons=4,
        noise_sigma=0.3,
        n_train=20,
        n_test=10,
        calibration_images=20,
        data_directory=tmp_path,
        download=False,
    )
    assert exp.x_train.shape == (20, 1, 64, 64) and exp.x_test.shape == (10, 1, 64, 64)
    assert exp.r_train.shape == exp.c_train.shape == (
        20,
        4,
    ) and exp.r_test.shape == exp.c_test.shape == (10, 4)
    assert exp.ceiling.shape == (4,) and (exp.ceiling < 1).all()
    images_ = torch.cat([exp.x_train, exp.x_test])
    teacher = GaborTeacher(4, seed=0).calibrate(images_[:20])
    noisy, clean = make_synthetic_neural_data(teacher, images_, noise_sigma=0.3, seed=1)
    torch.testing.assert_close(torch.cat([exp.r_train, exp.r_test]), noisy)
    torch.testing.assert_close(torch.cat([exp.c_train, exp.c_test]), clean)
    teacher_, x_train, *_ = exp  # a NamedTuple unpacks
    assert teacher_ is exp.teacher
    with pytest.raises(ValueError, match="validation set has 30"):
        make_synthetic_experiment(
            n_neurons=4, n_train=25, n_test=10, data_directory=tmp_path, download=False
        )
