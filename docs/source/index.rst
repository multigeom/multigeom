multigeom
=========

**Multi-scale information geometry of neural population codes, in Python.**

``multigeom`` computes the Fisher information metric :math:`J(x)` of any fitted PyTorch model of neural responses, and the
*multi-scale Fisher metric*

.. math::

   G(x) = \int_0^\infty [\phi_t * J_t](x)\,dt ,

the Fisher information of the stimulus-diffused likelihood integrated over all diffusion scales. :math:`G(x)` is exactly
related to the mutual information between stimulus and response: stimulus directions that are well encoded are expanded,
poorly encoded ones are contracted. For natural images it is estimated with a conditional diffusion model, and its leading
eigenvectors show which image features a neural population encodes best. The package also reproduces the macaque V1 and V4
analyses of the paper.

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc

      ``pip install multigeom`` and the optional extras.

   .. grid-item-card:: Usage
      :link: usage
      :link-type: doc

      Check an encoder, compute :math:`J(x)` and :math:`G(x)`, run the pipeline.

   .. grid-item-card:: Demos
      :link: demos
      :link-type: doc

      Notebooks from "bring your own encoder" to the figures of the paper.

Quick start
-----------

.. code-block:: python

   import multigeom

   multigeom.validate_encoder(encoder)                       # checks the encoder contract
   fisher = multigeom.compute_fisher(image=x, encoder=encoder, sigma=0.22, top_k=10)
   fisher.eigenvalues, fisher.eigenvectors                   # leading eigenpairs of J(x)

Links
-----

* `Landing page <https://multigeom.github.io>`_
* `Paper (arXiv:2605.06304) <https://arxiv.org/abs/2605.06304>`_
* `Source code <https://github.com/multigeom/multigeom>`_ and `tutorials <https://github.com/multigeom/multigeom-tutorials>`_
* `Paper code <https://github.com/sazio/MultiscaleInformationGeometry>`_, the frozen record of the published results

Licensing
---------

The code is released under the MIT license. The datasets used in the paper keep their own licenses.

Authors
-------

Simone Azeglio, Steeve Laquitaine, Ulisse Ferrari and Matthew Chalk
(Institut de la Vision, INSERM, CNRS, Sorbonne Université).

Key reference
-------------

Azeglio, S., Laquitaine, S., Ferrari, U., Chalk, M. *A multi-scale information geometry reveals the structure of mutual
information in neural populations.* Advances in Neural Information Processing Systems (NeurIPS), 2026.

.. code-block:: bibtex

   @inproceedings{azeglio2026multiscale,
     title     = {A multi-scale information geometry reveals the structure of mutual information in neural populations},
     author    = {Azeglio, Simone and Laquitaine, Steeve and Ferrari, Ulisse and Chalk, Matthew},
     booktitle = {Advances in Neural Information Processing Systems},
     year      = {2026}
   }

.. toctree::
   :hidden:
   :maxdepth: 2

   Home <self>
   Installation <installation>
   Usage <usage>
   Demos <demos>
   Contributing <contributing>
   API Docs <api>
