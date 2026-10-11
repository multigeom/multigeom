Installation
============

``multigeom`` needs Python 3.10 to 3.12.

From PyPI
---------

.. code-block:: bash

   pip install multigeom

This installs the core: the Fisher metric :math:`J(x)`, the :math:`G(x)` estimator, the encoder contract, the diffusion model
and the pipeline (PyTorch, torchvision, diffusers, NumPy, SciPy). Optional features come as extras:

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Extra
     - Adds
     - Install
   * - ``vision``
     - The ON-OFF reference encoder (``plenoptic``). Inception and ResNet-50 come with torchvision.
     - ``pip install "multigeom[vision]"``
   * - ``fit``
     - Fitting an encoder to neural data: TVSD loaders, layer sweep, noise calibration.
     - ``pip install "multigeom[fit]"``
   * - ``viz``
     - The figures of the paper (matplotlib).
     - ``pip install "multigeom[viz]"``
   * - ``modal``
     - Running the :math:`G(x)` estimation on serverless GPUs.
     - ``pip install "multigeom[modal]"``
   * - ``notebooks``
     - Jupyter kernel and notebook widgets.
     - ``pip install "multigeom[notebooks]"``
   * - ``all``
     - Everything above.
     - ``pip install "multigeom[all]"``

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
create the Jupyter kernel: see `multigeom-tutorials <https://github.com/steevelaquitaine/multigeom-tutorials>`_.
