# Changelog

All notable changes to `multigeom` are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/), and the
version numbers follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0.dev3] - 2026-10-15

### Added
- `multigeom.datasets.make_synthetic_experiment()` simulates a whole experiment in one call (TinyImageNet images, V1-like Gabor neurons, noise, the
  split for fitting and testing and the noise ceiling), returned as a `SyntheticExperiment`.
- `multigeom.datasets.download_tinyimagenet()` downloads and unpacks TinyImageNet (240 MB), once; the demo notebooks 1 and 2 call it, so
  they run on Google Colab without a shell script.
- `download_tvsd()` (237 MB, GIN) and `download_things_images(accept_license=True)` (5 GB, OSF; the licence is shown and must be accepted),
  both skipped when the data is already there; an interrupted download resumes.

### Changed
- `multigeom.data` is `multigeom.datasets`; the encoder contract `multigeom.encoder` is `multigeom.encoders.contract` (the names
  `validate_encoder`, `EncoderContractError` and `EncoderReport` are unchanged in `multigeom` and `multigeom.encoders`).
- `multigeom.viz` is removed: the scripts that draw the figures and the table of the paper are now the `figures/` folder of
  multigeom-tutorials. What was new moved into the package under a name that says what it does:
  `multigeom.encoders.receptive_field.readout_receptive_field`, `multigeom.analysis.match_eigenvectors`, and
  `multigeom.utils.show_figure` / `figure_to_png`. `multigeom.pipeline.metric_field_job.plot_metric_field` is removed.
- Lines of at most 88 characters in the package, the tests and the docs, formatted with `ruff format` and checked with `ruff check` (`[tool.ruff]` in `pyproject.toml`, and a *format* job in the tests workflow).

### Fixed
- `register_backbone` and `register_readout` accept a class defined again with the same name, module and qualified name, so a
  notebook cell that defines a registered class can be run twice. A different class with a taken name is still refused.

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
