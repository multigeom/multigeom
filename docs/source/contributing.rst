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

The build treats warnings as errors. The API pages come from the docstrings (Google style), so document new public functions
with ``Args:`` and ``Returns:`` sections.

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
3. Tag and push: ``git tag v1.0.0 && git push origin v1.0.0``. The *release* workflow checks that the tag equals the version, builds
   the sdist and the wheel, and publishes them to PyPI.

A version published to PyPI cannot be replaced or reused: fix a mistake with a new version.

One-time setup, on pypi.org and test.pypi.org: create a *pending publisher* for the project ``multigeom`` (owner ``multigeom``,
repository ``multigeom``, workflow ``release.yml``, environment ``pypi`` or ``testpypi``), and create the two environments in the
repository settings. No token is stored.
