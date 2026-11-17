Demo Notebooks
==============

We provide demo notebooks in three groups. Foundations computes each quantity on one neuron and a one-dimensional stimulus.
Getting started applies the package to encoders of images. Reproduce the paper redraws the figures of the paper. To run them you
need a working Jupyter notebook server and a ``multigeom`` installation. Click a notebook to read it with its stored outputs.

Foundations
-----------

Eight notebooks that build the posterior, the mutual information, the Fisher information :math:`J(x)` and the multi-scale Fisher
information :math:`G(x)` on one neuron. Its response to a stimulus :math:`x` on :math:`[-3, 3]` is its mean response at :math:`x`
(the tuning curve) plus Gaussian noise. Every quantity is computed exactly on a grid by ``multigeom.lowdim``. The notebooks need
no dataset, no diffusion model and no GPU, and each runs in seconds on a CPU.

.. nbgallery::
   :name: demo-gallery-foundations

   demo_notebooks/f1_what_one_response_tells_you
   demo_notebooks/f2_entropy_and_mutual_information
   demo_notebooks/f3_fisher_information
   demo_notebooks/f4_from_fisher_to_geometry
   demo_notebooks/f5_noise_as_a_ruler
   demo_notebooks/f6_which_stimuli_carry_the_information
   demo_notebooks/f7_multiscale_fisher_information
   demo_notebooks/f8_geometry_gives_back_the_information

Getting started
---------------

Three notebooks that take you from an encoder that already exists, to an encoder of your own, to the multi-scale metric. They need no
dataset beyond TinyImageNet (notebooks 1 and 2) and run in minutes on a GPU.

.. nbgallery::
   :name: demo-gallery-getting-started

   demo_notebooks/0_use_a_builtin_encoder
   demo_notebooks/1_create_your_encoder
   demo_notebooks/2_compute_g

Reproduce the paper
-------------------

These notebooks need the datasets, the trained checkpoints of the paper and a GPU.

.. nbgallery::
   :name: demo-gallery-paper

   demo_notebooks/3_leading_eigenvectors_of_G
   demo_notebooks/4_subspace_alignment
   demo_notebooks/5_local_metric_field
   demo_notebooks/6_bures_wasserstein

.. list-table::
   :header-rows: 1
   :widths: 38 34 28

   * - Notebook
     - What it does
     - Needs
   * - F1 What one response tells you
     - the response distribution :math:`p(r \mid x)` and the posterior :math:`p(x \mid r)` of one response, ``posterior``
     - a CPU, seconds
   * - F2 Entropy and mutual information
     - the bits removed by one response, :math:`I(R; X)` of three neurons, ``entropy``, ``mutual_information``
     - a CPU, seconds
   * - F3 Fisher information
     - :math:`J(x)` with ``compute_fisher``, the score, the Cramér-Rao bound
     - a CPU, seconds
   * - F4 From Fisher to geometry
     - Fisher distances, the :math:`2 \times 2` Fisher matrix, ``cumulative_distance``, ``fisher_from_jacobian``
     - a CPU, seconds
   * - F5 Noise as a ruler
     - blur, the best guess of a stimulus from its blurred version, the information at each blur level
     - a CPU, seconds
   * - F6 Which stimuli carry the information?
     - the blurred code, the local information :math:`i(x)`, a test of locality, ``blurred_code``, ``compute_g``
     - a CPU, seconds
   * - F7 The multi-scale Fisher information
     - :math:`G(x)` of three neurons with ``compute_g``, the contribution of each blur level, G distances
     - a CPU, seconds
   * - F8 The geometry gives back the information
     - :math:`I(R; X) = \tfrac{1}{2}\,\mathbb{E}[\operatorname{Tr} G]`, ``gaussian_g``, ``nearest_neighbour_information``
     - a CPU, seconds
   * - 0 Use an encoder that is already fitted
     - load an encoder of the paper, predict responses, compute :math:`J(x)`
     - a CPU, a minute (Colab works)
   * - 1 Build your own encoder
     - synthetic neural data, your own backbone and readout, ``fit_readout``, ``save``, ``load``, how to contribute it
     - TinyImageNet, about 1 min on a GPU
   * - 2 Train the diffusion model and compute G(x)
     - ``precompute_responses``, ``train_diffusion``, ``compute_g``
     - TinyImageNet, about 2 min on a GPU
   * - 3 Leading eigenvectors of G(x)
     - Fig. 3D-F and Fig. 4B-C
     - data, checkpoints, a GPU, about 2 min
   * - 4 Subspace alignment
     - Fig. 4D-E
     - data, checkpoints, GPUs (local or Modal), 1 min to hours
   * - 5 Local metric field
     - Fig. 5
     - data, checkpoints, a GPU, about 25 min
   * - 6 Bures-Wasserstein distance
     - Fig. 6 (appendix)
     - the outputs of notebook 4, a CPU, minutes

Installation
------------

See the :doc:`installation` page. The notebooks have their own repository with a conda environment file for the Jupyter kernel:
`multigeom-tutorials <https://github.com/multigeom/multigeom-tutorials>`_.

Demo data
---------

TinyImageNet (notebooks 1 and 2) and the other datasets (TVSD, THINGS images) and the trained checkpoints (notebooks 3 to 6) are described,
with download scripts, in the `multigeom-tutorials README <https://github.com/multigeom/multigeom-tutorials#data-and-checkpoints-notebooks-1-to-6>`_.
The checkpoints are on `Hugging Face <https://huggingface.co/sazio/multigeom>`_.

Contributing
------------

Found a problem, or want to add a demo? See :doc:`contributing`.
