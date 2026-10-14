Contributing
============

Contributions are welcome: bug reports and ideas as `issues <https://github.com/multigeom/multigeom/issues>`_, code as pull
requests.

Development install
-------------------

.. code-block:: bash

   git clone https://github.com/multigeom/multigeom.git
   cd multigeom
   pip install -e ".[all,dev,docs]"      # dev: pytest; docs: Sphinx and its plugins

Tests
-----

.. code-block:: bash

   pytest tests -q --tb=line -m "not slow and not gpu and not pretrained"   # what CI runs on every pull request
   pytest tests -q --tb=line                                                # also the tests that need checkpoints

The unit tests use synthetic data and run on a CPU. Tests that run the notebooks and compare their figures with stored
references are in the `multigeom-tutorials <https://github.com/multigeom/multigeom-tutorials>`_ repository.

Documentation
-------------

.. code-block:: bash

   cd docs
   make docs TUTORIALS_DIR=../../multigeom-tutorials     # a clone of multigeom-tutorials next to this repository
   python -m http.server -d build/html 8000

To preview the pages while you edit, use ``make livehtml`` instead. It serves the documentation at http://127.0.0.1:8000, rebuilds
when you save a ``.rst`` file or a docstring of the package, and reloads the browser tab:

.. code-block:: bash

   cd docs
   make livehtml TUTORIALS_DIR=../../multigeom-tutorials

The demo notebooks are copied when the command starts, so restart it after you change a notebook. After a change of the theme or
of ``conf.py``, stop it and run ``make clean`` first. Before you push, run ``make docs``: it is the clean build that CI runs.

The build treats warnings as errors. The API pages come from the docstrings (Google style), so document new public functions
with ``Args:`` and ``Returns:`` sections.

.. _add-an-encoder:

Add an encoder
--------------

A new backbone, readout or complete encoder joins the package through a pull request. Write and try it first in a notebook (the
section "Write your own backbone and readout" of the demo notebook 1 and the :doc:`usage page <usage>` give a working example), then:

1. Fork ``multigeom/multigeom``, clone your fork and branch from ``develop``:
   ``git switch -c encoder/my-encoder origin/develop``. Install with ``pip install -e ".[all,dev]"``.
2. Move the classes into the package, next to the existing ones: a backbone into ``multigeom/encoders/backbones.py``, a readout
   into ``multigeom/encoders/readout.py``. Keep the ``@register_backbone("name")`` or ``@register_readout("name")`` decorator
   (imported from ``.registry``), give every constructor argument a default (the saved architecture records the arguments), and
   write a Google-style docstring with ``Args:`` and ``Returns:`` that says what the model is and cites its source.
3. Make it importable from ``multigeom.encoders``. In ``multigeom/encoders/__init__.py``, add the class name to the import line of
   the module it lives in, and to the ``__all__`` list. For a backbone called ``GaborBankBackbone``:

   .. code-block:: python

      # Add the name to the existing import ...
      from .backbones import GaborBankBackbone, InceptionBackbone, ...
      # ... and to the list of public names.
      __all__ = [..., "GaborBankBackbone", ...]

   Users can then write ``from multigeom.encoders import GaborBankBackbone``. The import also matters for saved encoders: a saved
   file stores only the registered names (``"gabor_bank"``), and ``NeuralEncoder.load`` finds the class by importing
   ``multigeom.encoders``, which imports your module and runs its ``@register_backbone`` decorator. A class placed in a new
   module is registered only if ``__init__.py`` imports that module.
4. Add a unit test in ``tests/unit``. It must run on a CPU without data or downloads:

   .. code-block:: python

      def test_my_encoder_round_trips(tmp_path):
          encoder = NeuralEncoder(
              MyBackbone(), n_neurons=5, readout="my_readout")
          validate_encoder(encoder)             # the encoder contract
          x = torch.rand(4, 1, 64, 64) * 2 - 1
          assert encoder.predict(x).shape == (4, 5)    # 1 per neuron
          path = encoder.save(tmp_path / "e.pt")
          loaded = NeuralEncoder.load(path)
          torch.testing.assert_close(
              loaded.predict(x), encoder.predict(x))

5. Run the tests as in the section above, add a line to ``CHANGELOG.md``, and build the documentation (warnings are errors).
6. Push to your fork and open a pull request against ``develop``. Tests, build and docs must pass before review.

A complete encoder with its own ``forward`` (not just a new backbone or readout) can go in a new module of ``multigeom/encoders/``,
exported the same way. A model that needs trained weights or a large dataset is shared as a checkpoint instead: open an issue, and the weights can be
hosted next to the paper's encoders and loaded with ``load_encoder``.

Pull requests
-------------

* Branch from ``develop`` and open the pull request against ``develop``.
* Add or update tests for what you change, and keep the unit tests passing.
* Keep commits focused, with messages that say why the change is made.

Releases
--------

The version is set once, in ``multigeom/__init__.py`` (``__version__``), and ``pyproject.toml`` reads it. To release:

1. Update the version and ``CHANGELOG.md``, and merge to ``main`` once the tests and the docs build pass.
2. Try the release on TestPyPI: in the Actions tab, run the *release* workflow by hand. Install from there in a fresh environment
   (``pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ multigeom``) and import it.
3. Tag and push: ``git tag v0.1.0.dev1 && git push origin v0.1.0.dev1``. The *release* workflow checks that the tag equals the version, builds
   the sdist and the wheel, and publishes them to PyPI.

A version published to PyPI cannot be replaced or reused: fix a mistake with a new version.

One-time setup, on pypi.org and test.pypi.org: create a *pending publisher* for the project ``multigeom`` (owner ``multigeom``,
repository ``multigeom``, workflow ``release.yml``, environment ``pypi`` or ``testpypi``), and create the two environments in the
repository settings. No token is stored.
