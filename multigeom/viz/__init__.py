"""
Figures (extra `[viz]`, needs matplotlib): the figure panels of the paper (Figs. 3-6) and showing figures in a
notebook. Importing `multigeom.viz.style` switches matplotlib to the non-interactive Agg backend; use `show_figure` to
display a figure in a notebook.
"""
__all__ = ["show_figure", "figure_to_png", "plot_alignment", "plot_bures_wasserstein"]

_LAZY = {"show_figure": "display", "figure_to_png": "display", "plot_alignment": "alignment",
         "plot_bures_wasserstein": "alignment"}


def __getattr__(name):                       # matplotlib is imported only when a figure function is asked for
    if name in _LAZY:
        from importlib import import_module
        return getattr(import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
