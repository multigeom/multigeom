Contributing
============

Contributions are welcome: bug reports and ideas as `issues <https://github.com/multigeom/multigeom/issues>`_, code as pull
requests.

Development install
-------------------

.. code-block:: bash

   git clone https://github.com/multigeom/multigeom.git
   cd multigeom
   pip install -e ".[all,dev,docs]"

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

The version is set in ``multigeom/__init__.py``. Maintainers tag a release, and the package is published to PyPI.
