Demos
=====

The notebooks live in the `multigeom-tutorials <https://github.com/steevelaquitaine/multigeom-tutorials>`_ repository, with
the data download scripts and the tests that run them. They are shown here with their stored outputs.

Start here
----------

.. toctree::
   :maxdepth: 1

   demo_notebooks/0_bring_your_own_encoder

Check your own PyTorch model against the encoder contract and compute its Fisher metric. It runs in seconds on a CPU and
can be opened in Colab.

Reproduce the paper
-------------------

These notebooks need the datasets, the trained checkpoints and a GPU. The repository's README explains how to get them.

.. toctree::
   :maxdepth: 1

   demo_notebooks/1_leading_eigenvectors_of_G
   demo_notebooks/2_subspace_alignment
   demo_notebooks/3_local_metric_field
   demo_notebooks/4_bures_wasserstein

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - Notebook
     - Reproduces
     - Time
   * - Leading eigenvectors of G(x)
     - Fig. 3D-F and Fig. 4B-C
     - about 2 min
   * - Subspace alignment
     - Fig. 4D-E
     - 1 min to hours (local GPUs or Modal)
   * - Local metric field
     - Fig. 5
     - about 25 min on one GPU
   * - Bures-Wasserstein distance
     - Fig. 6 (appendix)
     - minutes on a CPU
