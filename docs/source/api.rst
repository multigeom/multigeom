API Docs
========

Public API
----------

The functions and classes of each group. The main ones are also available directly from ``multigeom`` (for example
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
   multigeom.geometry.compute_fisher_per_cell
   multigeom.geometry.compute_gaussian_env
   multigeom.geometry.compute_prior_variances
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
   multigeom.analysis.align_signs
   multigeom.analysis.compute_normalized
   multigeom.analysis.compute_information
   multigeom.analysis.compute_bootstrap
   multigeom.analysis.compute_band

Plotting
~~~~~~~~

The figures of the estimates at each noise scale (needs matplotlib; importing
``multigeom.plotting`` sets the style of the paper).

.. autosummary::
   :toctree: api/generated

   multigeom.plotting.plot_per_scale
   multigeom.plotting.plot_env_spectrum
   multigeom.plotting.plot_coding_spectrum

Module reference
----------------

Every module of the package, with its functions and classes.

.. autosummary::
   :toctree: api/modules
   :recursive:

   multigeom
