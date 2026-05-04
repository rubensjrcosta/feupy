# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Marker plotting utilities."""

from ..palettes import PALETTE_DEFAULT
from .resolver import resolve_marker
from .marker_size import resolve_marker_size


__all__ = [
    "build_point_kwargs",
     "build_fp_kwargs",
]


def build_point_kwargs(
    sources,
    marker_size=6,
    palette=None,
    uniform_size=True,
):
    """
    Build matplotlib marker kwargs for a collection of sources.

    Parameters
    ----------
    sources : list
        Iterable of sources or labels.
    marker_size : float
        Base marker size.
    palette : list, optional
        Color palette.
    uniform_size : bool
        Normalize marker sizes.

    Returns
    -------
    dict
        Mapping {label -> kwargs}.
    """

    palette = palette or PALETTE_DEFAULT

    result = {}

    for i, source in enumerate(sources):

        label = str(source)

        marker = resolve_marker(label)

        size = resolve_marker_size(marker, marker_size, uniform_size)

        color = palette[i % len(palette)][0]

        result[label] = dict(
            label=label,
            marker=marker,
            markersize=size,
            color=color,
            #markeredgecolor="black",
            #mew=0.4,
        )

    return result


def build_fp_kwargs(
    labels,
    sources=None,
    marker_size=6,
    palette=None,
    uniform_size=True,
):
    """
    Build plotting kwargs for spectral datasets.

    Marker is source-level.
    Color is dataset-level.

    Parameters
    ----------
    labels : list of str
        Dataset labels.
    sources : list, optional
        Source labels corresponding to each dataset.
    marker_size : float
        Base marker size.
    palette : list, optional
        Color palette.
    uniform_size : bool
        Normalize marker sizes.

    Returns
    -------
    dict
        Mapping {source_label -> kwargs}.
    """

    palette = palette or PALETTE_DEFAULT

    kwargs_dict = {}

    for i, label in enumerate(labels):

        source_label = str(sources[i]) if sources else label

        marker = resolve_marker(source_label)

        size = resolve_marker_size(marker, marker_size, uniform_size)

        color = palette[i % len(palette)][0]

        kwargs_dict[source_label] = dict(
            label=label,
            marker=marker,
            markersize=size,
            color=color,
            ls="solid",
        )

    return kwargs_dict