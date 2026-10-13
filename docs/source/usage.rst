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

   encoder = load_encoder("inception", area="V1")   # an encoder of the paper, downloaded once
   print(encoder.summary())

   images = ...                                     # [N, 1, 64, 64] tensor, values in [-1, 1]
   responses = encoder.predict(images)              # [N, 50] predicted neural responses

   # A checkpoint on disk works too:
   encoder = load_encoder("path/to/encoder.pt")

**Build your own.** Choose a backbone, a readout, fit the readout to neural data, save:

.. code-block:: python

   from multigeom import NeuralEncoder
   from multigeom.encoders import SmallCNNBackbone

   backbone = SmallCNNBackbone()                    # train it first, on any task (plain PyTorch)
   encoder = NeuralEncoder(backbone, n_neurons=50, readout="gaussian")

   # Freezes the backbone, fits the readout, and keeps the best epoch.
   report = encoder.fit_readout(images, responses)  # images [N, 1, 64, 64], responses [N, 50]

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

Your own backbone or readout needs the decorator ``@register_backbone("name")`` or ``@register_readout("name")`` so that a saved encoder
can be rebuilt from its file. A backbone is an ``nn.Module`` with attributes ``out_channels`` and ``out_spatial`` that maps images to
``[B, C, S, S]``.

**Any other PyTorch model** works too, without ``NeuralEncoder``, if it follows the contract below:

.. code-block:: python

   import multigeom

   multigeom.validate_encoder(model)                # raises an error naming the rule that is broken
   fisher = multigeom.compute_fisher(image=x, encoder=model, sigma=0.22, top_k=10)

The :doc:`demo notebooks <demos>` go through each step, with synthetic neural data that has a known ground truth.

The encoder contract
--------------------

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
