from functools import partial
from pathlib import Path


def save_fig(fig, fig_name, fig_dir, fig_fmt="png", fig_size=(6.4, 4), dpi=300, transparent_png=False):
    """Save an existing figure without introducing task-specific drawing code."""
    destination = Path(fig_dir) / fig_fmt
    destination.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(fig_size, forward=False)
    path = destination / f"{fig_name}.{fig_fmt}"
    fig.savefig(path, bbox_inches="tight", dpi=dpi, transparent=transparent_png if fig_fmt == "png" else False)
    return path


def setup_savefig(res_path, fig_fmt="png", dpi=300, transparent_png=False):
    return partial(save_fig, fig_dir=res_path, fig_fmt=fig_fmt, dpi=dpi, transparent_png=transparent_png)
