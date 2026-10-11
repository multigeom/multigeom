Demo Notebooks
==============

We provide a set of demo notebooks to get started with ``multigeom`` and to reproduce the figures of the paper. To run them
you need a working Jupyter notebook server, a ``multigeom`` installation, and, for notebooks 1 to 4, the datasets, the trained
checkpoints and a GPU. Click a notebook to read it with its stored outputs.

.. nbgallery::
   :name: demo-gallery
   :glob:

   demo_notebooks/0_bring_your_own_encoder
   demo_notebooks/1_leading_eigenvectors_of_G
   demo_notebooks/2_subspace_alignment
   demo_notebooks/3_local_metric_field
   demo_notebooks/4_bures_wasserstein

.. list-table::
   :header-rows: 1
   :widths: 38 34 28

   * - Notebook
     - Reproduces
     - Needs
   * - Bring your own encoder
     - the encoder contract and :math:`J(x)` for a model of your own
     - a CPU, a few seconds (Colab works)
   * - Leading eigenvectors of G(x)
     - Fig. 3D-F and Fig. 4B-C
     - data, checkpoints, a GPU, about 2 min
   * - Subspace alignment
     - Fig. 4D-E
     - data, checkpoints, GPUs (local or Modal), 1 min to hours
   * - Local metric field
     - Fig. 5
     - data, checkpoints, a GPU, about 25 min
   * - Bures-Wasserstein distance
     - Fig. 6 (appendix)
     - the outputs of notebook 2, a CPU, minutes

Installation
------------

See the :doc:`installation` page. The notebooks have their own repository with a conda environment file for the Jupyter kernel:
`multigeom-tutorials <https://github.com/multigeom/multigeom-tutorials>`_.

Demo data
---------

The datasets (TVSD, THINGS images, TinyImageNet) and the trained checkpoints are described, with download scripts, in the
`multigeom-tutorials README <https://github.com/multigeom/multigeom-tutorials#data-and-checkpoints-notebooks-1-to-4>`_. The
checkpoints are on `Hugging Face <https://huggingface.co/steevelaquitaine/multigeom>`_.

Contributing
------------

Found a problem, or want to add a demo? See :doc:`contributing`.
