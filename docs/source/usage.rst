Usage
=====

The package does three things: it gives you **encoders** (models that predict neural responses from images), it computes the
**geometry** of an encoder, and it offers **analyses** that compare geometries. These are the functions you call, by group.

.. list-table::
   :header-rows: 1
   :widths: 14 38 48

   * - Group
     - Function
     - What it does
   * - Encoders
     - :func:`multigeom.encoders.load_encoder`
     - Returns an encoder of the paper, fitted to macaque V1 or V4 (downloaded once).
   * - Encoders
     - :class:`multigeom.encoders.NeuralEncoder`
     - Builds your own encoder (backbone and readout). Methods: ``predict``, ``fit_readout``, ``save``, ``load``,
       ``load_checkpoint`` and ``summary``.
   * - Encoders
     - :func:`multigeom.encoders.validate_encoder`
     - Checks that any PyTorch model follows the encoder contract below, and names the rule it breaks.
   * - Geometry
     - :func:`multigeom.geometry.compute_fisher`
     - Top eigenvalues and eigenvectors of the Fisher information :math:`J(x)` of an image, for an encoder.
   * - Geometry
     - :func:`multigeom.geometry.compute_g`
     - Top eigenpairs of the multi-scale Fisher metric :math:`G(x)` of an image, for an encoder and its trained diffusion model
       (see ``multigeom.diffusion``: ``precompute_responses`` and ``train_diffusion``).
   * - Analysis
     - :func:`multigeom.analysis.subspace_alignment`
     - Subspace alignment of two sets of eigenvectors.
   * - Analysis
     - :func:`multigeom.analysis.bures_wasserstein`
     - Bures-Wasserstein distance between two reconstructed metrics.

The geometry functions take an encoder and an image. The analysis functions take the eigenvectors that the geometry functions return.
Each group is a subpackage (``multigeom.encoders``, ``multigeom.geometry``, ``multigeom.analysis``), and the main functions are also
available directly from ``multigeom``, for example ``multigeom.compute_fisher``.

Encoders: use one, build one
----------------------------

An encoder predicts neural responses from images. The package represents it as a :class:`~multigeom.encoders.NeuralEncoder`: a
**backbone** (a feature extractor) followed by a **readout** (the part fitted to neural data). Images are always
``[N, 1, 64, 64]`` tensors with values in ``[-1, 1]``.

**Use an encoder that is already fitted.** The encoders of the paper (``"inception"`` and ``"resnet50"``, areas ``"V1"`` and
``"V4"``) are downloaded from Hugging Face once and cached:

.. code-block:: python

   from multigeom import load_encoder

   # An encoder of the paper, downloaded once.
   encoder = load_encoder("inception", area="V1")
   print(encoder.summary())

   images = ...                         # [N, 1, 64, 64], values in [-1, 1]
   responses = encoder.predict(images)  # [N, 50] predicted responses

   # A checkpoint on disk works too:
   encoder = load_encoder("path/to/encoder.pt")

**Build your own.** Choose a backbone, a readout, fit the readout to neural data, save:

.. code-block:: python

   from multigeom import NeuralEncoder
   from multigeom.encoders import SmallCNNBackbone

   # Train the backbone first, on any task (plain PyTorch).
   backbone = SmallCNNBackbone()
   encoder = NeuralEncoder(backbone, n_neurons=50, readout="gaussian")

   # Freezes the backbone, fits the readout, keeps the best epoch.
   # images [N, 1, 64, 64], responses [N, 50]
   report = encoder.fit_readout(images, responses)

   encoder.save("my_encoder.pt", area="V1")
   encoder = NeuralEncoder.load("my_encoder.pt")

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Readout
     - Each neuron computes
   * - ``"linear"``
     - a weighted sum of the feature channels, averaged over space
   * - ``"gaussian"``
     - the same over a Gaussian window of its own (position and width learned): the readout of the paper
   * - ``"factorized"``
     - the same over a window of any shape (one learned weight per location)

Write your own backbone and readout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both parts are ordinary ``nn.Module`` classes. A backbone maps images ``[B, 1, 64, 64]`` to a feature map ``[B, C, S, S]`` and has
the attributes ``out_channels`` (C) and ``out_spatial`` (S). A readout maps that feature map to responses ``[B, N]`` and is built
as ``(n_neurons, n_channels, spatial, dropout)``. The decorators ``@register_backbone("name")`` and ``@register_readout("name")``
give each class a name (see below for what the name is used for). This example fits an encoder to synthetic neural data from
start to finish:

.. code-block:: python

   import math
   import torch, torch.nn as nn, torch.nn.functional as F
   from multigeom import NeuralEncoder, validate_encoder
   from multigeom.data import GaborTeacher, make_synthetic_neural_data
   from multigeom.encoders import register_backbone, register_readout


   @register_backbone("gabor_bank")
   class GaborBankBackbone(nn.Module):
       """Energy of oriented Gabor filters. Nothing to train."""

       out_spatial = 16                   # 64 x 64 pixels, pooled by 4

       def __init__(self, n_orientations: int = 8, size: int = 11,
                    wavelength: float = 6.0):
           super().__init__()
           self.out_channels = 2 * n_orientations      # 2 phases each
           r = torch.arange(size) - size // 2
           yy, xx = torch.meshgrid(r, r, indexing="ij")
           env = torch.exp(-(xx**2 + yy**2) / (2 * (size / 4) ** 2))
           filters = []
           for k in range(n_orientations):
               a = math.pi * k / n_orientations
               u = xx * math.cos(a) + yy * math.sin(a)
               for phase in (0.0, math.pi / 2):
                   g = env * torch.cos(2 * math.pi * u / wavelength + phase)
                   filters.append(g - g.mean())
           w = torch.stack(filters)[:, None]
           norm = w.flatten(1).norm(dim=1)[:, None, None, None]
           self.register_buffer("filters", w / norm)

       def forward(self, x):
           out = F.conv2d(x, self.filters, padding="same") ** 2
           return F.avg_pool2d(out, 4)


   @register_readout("softplus")
   class SoftplusReadout(nn.Module):
       """Spatial mean, one linear map, positive responses."""

       def __init__(self, n_neurons: int, n_channels: int, spatial: int,
                    dropout: float = 0.0):
           super().__init__()
           self.linear = nn.Linear(n_channels, n_neurons)
           self.dropout = nn.Dropout(dropout)

       def forward(self, features):       # [B, C, S, S] -> [B, N]
           x = self.dropout(features).mean(dim=(2, 3))
           return F.softplus(self.linear(x))


   # Smooth random images stand in for natural ones (use TinyImageNet).
   low = torch.randn(1200, 1, 8, 8)
   images = F.interpolate(low, size=64, mode="bicubic").clamp(-1, 1)

   # Synthetic neurons with a known ground truth.
   teacher = GaborTeacher(20, seed=0).calibrate(images)
   noisy, clean = make_synthetic_neural_data(
       teacher, images, noise_sigma=0.22, seed=1)

   encoder = NeuralEncoder(GaborBankBackbone(), n_neurons=20,
                           readout="softplus")
   # Fits the readout; the backbone stays fixed.
   encoder.fit_readout(images[:1000], noisy[:1000], epochs=60)
   responses = encoder.predict(images[1000:])      # [200, 20]
   validate_encoder(encoder)                       # the encoder contract

   encoder.save("my_encoder.pt")
   print(encoder.config["backbone"]["name"],       # gabor_bank
         encoder.config["readout"]["name"])        # softplus

**What the names are for.** A registered name is used in two places:

* *Building an encoder from a name.* A readout can be given by its name, ``readout="softplus"``, as above. A backbone is passed as
  an object (``GaborBankBackbone()``), so its name is not used when you build the encoder.
* *Saving and loading.* ``encoder.save`` stores the name and the constructor arguments of the backbone and the readout next to
  the weights. ``NeuralEncoder.load`` reads them and looks the names up in the registry to rebuild the encoder, so it needs no
  other code. This works in the session above because both classes are registered. In a new Python session they are registered
  only once their code has run, by importing the module that defines them:

.. code-block:: python

   from multigeom import NeuralEncoder
   import my_encoders       # registers "gabor_bank" and "softplus"
   # Without the import: "the backbone 'gabor_bank' is not registered"
   encoder = NeuralEncoder.load("my_encoder.pt")

The registries are ``multigeom.encoders.BACKBONES`` and ``READOUTS`` (dictionaries from name to class). A class that is part of
``multigeom`` itself is registered as soon as ``multigeom.encoders`` is imported, which is why the encoders of the package need no
extra import.

To share a new backbone or readout with everyone, see :ref:`add-an-encoder`.

**Any other PyTorch model** works too, without ``NeuralEncoder``, if it follows the contract below:

.. code-block:: python

   import multigeom

   # Raises an error that names the rule that is broken.
   multigeom.validate_encoder(model)
   fisher = multigeom.compute_fisher(
       image=x, encoder=model, sigma=0.22, top_k=10)

The :doc:`demo notebooks <demos>` go through each step, with synthetic neural data that has a known ground truth.

The encoder contract
^^^^^^^^^^^^^^^^^^^^

``validate_encoder`` runs a few forward and backward passes on random images and raises
:class:`multigeom.EncoderContractError`, naming the rule and what to change, at the first violation.

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Rule
     - Requirement
   * - R1
     - an ``nn.Module``
   * - R2
     - input ``[B, 1, 64, 64]``, values in [-1, 1]; the model does any rescaling or channel repeat itself
   * - R3
     - output ``[B, N]``, floating point and finite, the same N for every batch size (and ``n_neurons == N`` if set)
   * - R4
     - deterministic in eval mode (no dropout or noise at inference)
   * - R5
     - differentiable with respect to the input through ``forward`` (no ``torch.no_grad()`` or ``detach()``): needed for :math:`J(x)`
   * - R6
     - samples independent within a batch (no batch statistics in train mode)
   * - R7
     - does not modify its input in place
   * - R8
     - responses vary across images (warning)
   * - R9
     - responses of order 1, for example z-scored, since the noise sigma is 0.22 (warning)

Training the diffusion model on the encoder's responses needs R1 to R4, R6 and R9; estimating :math:`G(x)` and :math:`J(x)`
needs all of them. The quality of the fit and the 64 x 64 grayscale natural-image domain are not checked. The diffusion
model's conditioning layer is sized by N, so each encoder needs its own diffusion model.

The pipeline
------------

The paper's analysis runs in stages, each a module with a command line (``python -m multigeom...`` or ``multigeom <command>``):

.. list-table::
   :header-rows: 1
   :widths: 5 40 55

   * -
     - Module
     - What it does
   * - 1
     - ``multigeom.fit.train_encoder``
     - Fit a readout on a frozen backbone to neural data (extra ``paper``).
   * - 2
     - ``multigeom.diffusion.responses``
     - Evaluate the encoder on TinyImageNet; these responses condition the diffusion model.
   * - 3
     - ``multigeom.diffusion.train``
     - Train the conditional diffusion model (resumable).
   * - 4
     - ``multigeom.pipeline.per_image``
     - :math:`G(x)` and :math:`J(x)` for a set of images (resumable, seeded per image).
   * - 5
     - ``multigeom.pipeline.alignment``
     - Subspace alignment and Bures-Wasserstein distance between encoders and areas.
   * - 6
     - ``multigeom.pipeline.metric_field``
     - The local metric field of paper Fig. 5.

.. code-block:: bash

   multigeom --help
   multigeom compute-g --help

Settings are YAML files packaged with the library (``estimator``, ``diffusion``, ``inception``, ``resnet50``, ``onoff``) and
loaded by name with :func:`multigeom.config.load_config`.

Working folders
---------------

The pipeline reads and writes ``runs/`` (checkpoints and per-image results) and ``data/`` (datasets) in the current directory.
Set ``MULTIGEOM_REPO`` to use another root, and ``NEURIPS_DATA_DIR`` to read the datasets from elsewhere.
