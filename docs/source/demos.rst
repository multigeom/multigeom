Demo Notebooks
==============

We provide a set of demo notebooks, from the basic blocks of the package to the figures of the paper. To run them you need a working
Jupyter notebook server and a ``multigeom`` installation. Click a notebook to read it with its stored outputs.

Getting started
---------------

Three notebooks that take you from an encoder that already exists, to an encoder of your own, to the multi-scale metric. They need no
dataset beyond TinyImageNet (notebooks 1 and 2) and run in minutes on a GPU.

.. nbgallery::
   :name: demo-gallery-getting-started

   demo_notebooks/0_getting_started_use_a_builtin_encoder
   demo_notebooks/1_getting_started_synthetic_neural_data
   demo_notebooks/2_getting_started_diffusion_and_pipeline

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
   * - 0 Use an encoder that is already fitted
     - load an encoder of the paper, predict responses, compute :math:`J(x)`
     - a CPU, a minute (Colab works)
   * - 1 Build your own encoder
     - synthetic neural data, backbone, readout, ``fit_readout``, ``save``, ``load``
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
The checkpoints are on `Hugging Face <https://huggingface.co/steevelaquitaine/multigeom>`_.

Contributing
------------

Found a problem, or want to add a demo? See :doc:`contributing`.
