Installation
============

``multigeom`` needs Python 3.10 to 3.12.

From PyPI
---------

.. code-block:: bash

   pip install multigeom

This installs the core: encoders, the Fisher metric :math:`J(x)`, the diffusion model, the :math:`G(x)` estimator, and the
information-geometric analyses. Two extras add the rest:

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Install
     - Adds
     - For
   * - ``pip install "multigeom[paper]"``
     - The ON-OFF encoder (``plenoptic``) and the loaders of the TVSD recordings (``h5py``).
     - reproducing the paper
   * - ``pip install "multigeom[all]"``
     - Everything in ``paper``, plus Jupyter and Modal (serverless GPUs).
     - the demo notebooks, everything

Check the installation:

.. code-block:: bash

   python -c "import multigeom; print(multigeom.__version__)"

GPU
---

Estimating :math:`G(x)` is about 15 s per image on a recent GPU and far slower on a CPU (about 15 min per image on 32 cores).
Computing :math:`J(x)` takes a fraction of a second and runs anywhere. Install the PyTorch build that matches your CUDA
driver first, following the `PyTorch instructions <https://pytorch.org/get-started/locally/>`_, then install ``multigeom``.

From source
-----------

.. code-block:: bash

   git clone https://github.com/multigeom/multigeom.git
   cd multigeom
   pip install -e ".[all]"

An editable install picks up your changes to the code without reinstalling.

Environment for the demos
-------------------------

The notebooks have their own repository, with a conda environment file that pins PyTorch for CUDA 12.8 and the steps to
create the Jupyter kernel: see `multigeom-tutorials <https://github.com/multigeom/multigeom-tutorials>`_.
