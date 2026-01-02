"""Plotting helpers for markers."""

import astropy.units as u
from feupy.visualization.styles.markers.defaults import MARKERS_DEFAULT_DICT
from feupy.visualization.styles.palettes import PALETTE_DEFAULT
from feupy.visualization.styles.linestyles import LINESTYLES_DEFAULT


from .core import repeat_items

def make_marker_dict(
    labels,
    marker="o",
    marker_size=6,
    palette=None,
):
    palette = palette or PALETTE_DEFAULT
    scale = MARKERS_DEFAULT_DICT[marker][0]

    return {
        label: dict(
            label=label,
            marker=marker,
            markersize=round(marker_size * scale, 2),
            color=palette[i][0],
        )
        for i, label in enumerate(labels)
    }


def get_fit_plot_kwargs(
    labels,
    energy_bounds=[[5e-2, 2e3] * u.TeV],
    color_cycle=False,
    marker=",",
):
    if len(energy_bounds) == 1:
        energy_bounds *= len(labels)

    linestyles = repeat_items(LINESTYLES_DEFAULT, len(labels))
    palette = repeat_items(PALETTE_DEFAULT, len(labels), shuffle=True)

    return {
        i: dict(
            label=label,
            energy_bounds=energy_bounds[i],
            ls=linestyles[i],
            marker=marker,
            color=palette[i][0] if color_cycle else "black",
        )
        for i, label in enumerate(labels)
    }
