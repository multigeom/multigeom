"""Showing matplotlib figures in a notebook whatever the matplotlib backend."""
from __future__ import annotations


def figure_to_png(fig, dpi: int = 110) -> bytes:
    """PNG bytes of a matplotlib figure, independent of the matplotlib backend in use."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


def show_figure(fig, dpi: int = 110) -> None:
    """
    Display a figure in a notebook as an image. Displaying the PNG works whatever the matplotlib backend
    (in the non-interactive Agg backend a figure is shown as the text "<Figure size ...>" instead of a picture).
    """
    import matplotlib.pyplot as plt
    from IPython.display import Image, display
    display(Image(data=figure_to_png(fig, dpi)))
    plt.close(fig)
