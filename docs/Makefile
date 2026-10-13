# Build the documentation:  make docs        (needs: pip install -e ".[docs]")
# Fetch the demo notebooks: make notebooks TUTORIALS_DIR=../multigeom-tutorials
# Preview while editing:    make livehtml   (rebuilds and reloads http://127.0.0.1:8000 on every save)
SPHINXOPTS    ?=
SOURCEDIR     = source
BUILDDIR      = build
TUTORIALS_DIR ?= ../../multigeom-tutorials

.PHONY: docs livehtml notebooks linkcheck clean

notebooks:
	mkdir -p $(SOURCEDIR)/demo_notebooks
	cp $(TUTORIALS_DIR)/notebooks/*.ipynb $(SOURCEDIR)/demo_notebooks/

docs: notebooks
	sphinx-build -W --keep-going -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html

# The notebooks are copied once, when the command starts. The generated API pages and the copies are ignored, otherwise
# writing them would trigger a new build in a loop. --watch ../multigeom rebuilds on a change of a docstring.
livehtml: notebooks
	sphinx-autobuild -W --keep-going $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/live --port 8000 \
	  --watch ../multigeom --ignore "$(SOURCEDIR)/api/generated/*" --ignore "$(SOURCEDIR)/demo_notebooks/*"

linkcheck: notebooks
	sphinx-build -b linkcheck $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/linkcheck

clean:
	rm -rf $(BUILDDIR) $(SOURCEDIR)/api/generated $(SOURCEDIR)/api/modules
