API Docs
========

Public API
----------

The functions and classes of the three groups. The main ones are also available directly from ``multigeom`` (for example
``multigeom.compute_fisher``).

Encoders
~~~~~~~~

.. autosummary::
   :toctree: api/generated

   multigeom.encoders.load_encoder
   multigeom.encoders.NeuralEncoder
   multigeom.encoders.validate_encoder
   multigeom.encoders.EncoderReport
   multigeom.encoders.EncoderContractError
   multigeom.Encoder

Geometry
~~~~~~~~

.. autosummary::
   :toctree: api/generated

   multigeom.geometry.compute_fisher
   multigeom.geometry.compute_g
   multigeom.geometry.compute_g_per_scale
   multigeom.geometry.compute_env_per_scale
   multigeom.geometry.compute_env_spectrum
   multigeom.geometry.compute_coding_spectrum
   multigeom.geometry.FisherOutput
   multigeom.geometry.EstimatorOutput
   multigeom.geometry.PerScaleOutput
   multigeom.geometry.SpectrumOutput
   multigeom.geometry.CodingSpectrumOutput

Analysis
~~~~~~~~

.. autosummary::
   :toctree: api/generated

   multigeom.analysis.subspace_alignment
   multigeom.analysis.bures_wasserstein

Module reference
----------------

Every module of the package, with its functions and classes.

.. autosummary::
   :toctree: api/modules
   :recursive:

   multigeom
