# Build the documentation:  make docs        (needs: pip install -e ".[docs]")
# Fetch the demo notebooks: make notebooks TUTORIALS_DIR=../multigeom-tutorials
SPHINXOPTS    ?=
SOURCEDIR     = source
BUILDDIR      = build
TUTORIALS_DIR ?= ../../multigeom-tutorials

.PHONY: docs notebooks linkcheck clean

notebooks:
	mkdir -p $(SOURCEDIR)/demo_notebooks
	cp $(TUTORIALS_DIR)/notebooks/*.ipynb $(SOURCEDIR)/demo_notebooks/

docs: notebooks
	sphinx-build -W --keep-going -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html

linkcheck: notebooks
	sphinx-build -b linkcheck $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/linkcheck

clean:
	rm -rf $(BUILDDIR) $(SOURCEDIR)/api/generated $(SOURCEDIR)/api/modules
