"""Sphinx configuration of the multigeom documentation."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Heavy dependencies are mocked: the API pages are built from the docstrings, without installing PyTorch.
autodoc_mock_imports = [
    "torch", "torchvision", "diffusers", "plenoptic", "h5py", "matplotlib", "modal", "huggingface_hub",
    "transformers", "IPython",
]

import re  # noqa: E402

project = "multigeom"
author = "Simone Azeglio, Steeve Laquitaine, Ulisse Ferrari, Matthew Chalk"
copyright = "2026, the multigeom authors"
release = re.search(r'__version__ = "([^"]+)"', (ROOT / "multigeom" / "__init__.py").read_text()).group(1)
version = release

# Where the demo notebooks live (they are copied into demo_notebooks/ by `make notebooks`).
TUTORIALS_REPO = "multigeom/multigeom-tutorials"
TUTORIALS_BRANCH = "main"
LANDING_PAGE = "https://multigeom.github.io"
PAPER = "https://arxiv.org/abs/2605.06304"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "sphinx_design",
    "sphinx_copybutton",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
source_suffix = {".rst": "restructuredtext"}
master_doc = "index"

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# Notebooks: they need data and GPUs, so they are shown with the outputs stored in them, never executed here.
nbsphinx_execute = "never"
nbsphinx_prolog = r"""
{% set nb = env.doc2path(env.docname, base=None).split('/')[-1] %}
.. raw:: html

    <div class="nb-note">
      <a href="https://github.com/""" + TUTORIALS_REPO + r"""/blob/""" + TUTORIALS_BRANCH + r"""/notebooks/{{ nb }}">View on GitHub</a>
      {% if nb.startswith('0_') or nb.startswith('1_') or nb.startswith('2_') %}
      <a href="https://colab.research.google.com/github/""" + TUTORIALS_REPO + r"""/blob/""" + TUTORIALS_BRANCH + r"""/notebooks/{{ nb }}">Open in Colab</a>
      {% else %}
      Needs the datasets and trained checkpoints, and a GPU (see the notebook's first cells).
      {% endif %}
    </div>
"""

html_theme = "pydata_sphinx_theme"
html_title = "multigeom"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_logo = "_static/img/logo.svg"
html_favicon = "_static/img/logo.svg"
# Thumbnails of the demo grid (docs/source/demos.rst), by notebook.
nbsphinx_thumbnails = {
    "demo_notebooks/0_use_a_builtin_encoder": "_static/thumbnails/use_builtin_encoder.png",
    "demo_notebooks/1_create_your_encoder": "_static/thumbnails/create_your_encoder.png",
    "demo_notebooks/2_compute_g": "_static/thumbnails/diffusion_and_g.png",
    "demo_notebooks/3_leading_eigenvectors_of_G": "_static/thumbnails/leading_eigenvectors.png",
    "demo_notebooks/4_subspace_alignment": "_static/thumbnails/subspace_alignment.png",
    "demo_notebooks/5_local_metric_field": "_static/thumbnails/local_metric_field.png",
    "demo_notebooks/6_bures_wasserstein": "_static/thumbnails/bures_wasserstein.png",
}

# Navigation on the left only, as in the CEBRA docs: the section pages and the headings of the current page share the
# left sidebar, and the right-hand "on this page" column is switched off.
html_sidebars = {"**": ["sidebar-nav-bs", "page-toc"]}

html_theme_options = {
    "secondary_sidebar_items": [],
    "logo": {"text": ""},
    "header_links_before_dropdown": 6,
    "navbar_align": "left",
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "icon_links": [
        {"name": "Landing page", "url": LANDING_PAGE, "icon": "fa-solid fa-house"},
        {"name": "GitHub", "url": "https://github.com/multigeom/multigeom", "icon": "fa-brands fa-github"},
        {"name": "PyPI", "url": "https://pypi.org/project/multigeom/", "icon": "fa-brands fa-python"},
        {"name": "Paper", "url": PAPER, "icon": "fa-solid fa-file-lines"},
    ],
}


def _module_docstring_as_notes(app, what, name, obj, options, lines):
    """
    Module docstrings of the pipeline and figure modules are plain-text notes (algorithms, formulas, usage). Show the
    first paragraph as text and the rest verbatim, in a literal block, so that the layout of the notes is kept.
    """
    if what != "module" or not lines:
        return
    end = next((i for i, line in enumerate(lines) if not line.strip() and i > 0), len(lines))
    summary, rest = lines[:end], lines[end:]
    if not any(line.strip() for line in rest):
        return
    lines[:] = summary + ["", "::", ""] + ["   " + line if line.strip() else "" for line in rest]


def setup(app):
    app.connect("autodoc-process-docstring", _module_docstring_as_notes)
