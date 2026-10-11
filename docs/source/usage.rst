Usage
=====

The Python API has five entry points. Each takes an *encoder*: any fitted PyTorch model that maps images to neural responses
and follows the contract below.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - What it does
   * - :func:`multigeom.validate_encoder`
     - Checks that an encoder follows the contract and names the rule it breaks.
   * - :func:`multigeom.compute_fisher`
     - Top eigenvalues and eigenvectors of the Fisher information :math:`J(x)` of one image.
   * - :func:`multigeom.estimate_g`
     - Top eigenpairs of the multi-scale Fisher metric :math:`G(x)`, with a trained diffusion model.
   * - :func:`multigeom.align`
     - Subspace alignment of two sets of eigenvectors.
   * - :func:`multigeom.bures_wasserstein`
     - Bures-Wasserstein distance between two reconstructed metrics.

Bring your own encoder
----------------------

.. code-block:: python

   import torch, torch.nn as nn
   import multigeom

   class MyEncoder(nn.Module):
       def __init__(self, n_neurons=20):
           super().__init__()
           self.linear = nn.Linear(64 * 64, n_neurons)
       def forward(self, x):                                  # x: [B, 1, 64, 64] in [-1, 1]
           return torch.tanh(self.linear(x.flatten(1)))       # responses: [B, N]

   encoder = MyEncoder()
   report = multigeom.validate_encoder(encoder, sigma=0.22)   # raises EncoderContractError on a violation

   x = torch.rand(1, 1, 64, 64) * 2 - 1
   fisher = multigeom.compute_fisher(image=x, encoder=encoder, sigma=0.22, top_k=10)

The :doc:`demo notebook <demo_notebooks/0_bring_your_own_encoder>` runs this end to end.

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
     - Fit a readout on a frozen backbone to neural data (extra ``fit``).
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
   multigeom estimate-g --help

Settings are YAML files packaged with the library (``estimator``, ``diffusion``, ``inception``, ``resnet50``, ``onoff``) and
loaded by name with :func:`multigeom.config.load_config`.

Working folders
---------------

The pipeline reads and writes ``runs/`` (checkpoints and per-image results) and ``data/`` (datasets) in the current directory.
Set ``MULTIGEOM_REPO`` to use another root, and ``NEURIPS_DATA_DIR`` to read the datasets from elsewhere.
