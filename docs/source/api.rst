API Docs
========

Public API
----------

The functions and classes of the four groups. The main ones of the first three are also available directly from
``multigeom`` (for example ``multigeom.compute_fisher``).

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
   multigeom.geometry.FisherOutput
   multigeom.geometry.EstimatorOutput

Analysis
~~~~~~~~

.. autosummary::
   :toctree: api/generated

   multigeom.analysis.subspace_alignment
   multigeom.analysis.bures_wasserstein

Low-dimensional stimuli
~~~~~~~~~~~~~~~~~~~~~~~

One neuron and a one-dimensional stimulus on a grid, computed by quadrature without a diffusion model, and the linear
Gaussian code in any dimension. Reached as ``multigeom.lowdim``.

.. autosummary::
   :toctree: api/generated

   multigeom.lowdim.compute_fisher
   multigeom.lowdim.compute_g
   multigeom.lowdim.MultiscaleOutput
   multigeom.lowdim.blur_levels
   multigeom.lowdim.blurred_code
   multigeom.lowdim.mutual_information
   multigeom.lowdim.specific_surprise
   multigeom.lowdim.stimulus_specific_information
   multigeom.lowdim.distance
   multigeom.lowdim.gaussian_g
   multigeom.lowdim.gaussian_information
   multigeom.lowdim.nearest_neighbour_information
   multigeom.lowdim.to_bits

Module reference
----------------

Every module of the package, with its functions and classes.

.. autosummary::
   :toctree: api/modules
   :recursive:

   multigeom
