# Changelog

All notable changes to `multigeom` are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/), and the
version numbers follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `multigeom.data.download_tinyimagenet()` downloads and unpacks TinyImageNet (240 MB), once; the demo notebooks 1 and 2 call it, so
  they run on Google Colab without a shell script.

## [0.1.0.dev2] - 2026-10-14

Python 3.13 is supported (`requires-python` was `<3.13`, so `pip install` failed on Google Colab); the unit tests also run on 3.13.

## [0.1.0.dev1] - 2026-10-14

First pre-release on PyPI, to test the installation (for example on Google Colab) before the first stable release. It has everything
listed under 1.0.0 below. Install it with `pip install --pre multigeom`.

## [1.0.0] - unreleased

First release of the package, refactored from the code of the paper
*A multi-scale information geometry reveals the structure of mutual information in neural populations*.

### Added
- `NeuralEncoder` (backbone and readout) with `predict`, `fit_readout`, `save` and `load`; `load_encoder` for the encoders of the paper
  (downloaded from Hugging Face); the readouts `gaussian`, `linear` and `factorized`; the trainable `SmallCNNBackbone`;
  registries to add your own backbones and readouts.
- Documentation: a complete example of an encoder with a custom backbone and readout fitted to synthetic data, and a guide to
  contributing an encoder to the package.
- Synthetic neural data with a known ground truth (`GaborTeacher`, `make_synthetic_neural_data`, `noise_ceiling`).
- The Fisher information `compute_fisher`, the multi-scale metric `compute_g`, the encoder contract `validate_encoder`, the analyses
  `subspace_alignment` and `bures_wasserstein`. The functions are grouped in the subpackages `multigeom.encoders`, `multigeom.geometry`
  and `multigeom.analysis`, and the main ones are also available directly from `multigeom`.
- The conditional diffusion model with callable `train_diffusion` and `precompute_responses`; the pipeline for many images (resumable,
  local GPUs or Modal) and the command line `multigeom`.
- Documentation at <https://multigeom.github.io/multigeom/> and the demo notebooks in `multigeom-tutorials`.

### Changed
- One way to fit a readout: `NeuralEncoder.fit_readout`. The paper's script `multigeom.fit.train_encoder` and the layer sweep call
  it instead of keeping their own training loops, and `train-encoder` now writes the checkpoint with `NeuralEncoder.save`
  (the test scores are in its `metadata`). Older checkpoints of the paper load as before.
- Installation: the core, `multigeom[paper]` and `multigeom[all]`.
