# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Marker plotting utilities."""

from ..palettes import PALETTE_DEFAULT
from .marker_size import resolve_marker_size
from .resolver import resolve_marker

__all__ = [
    "build_fp_kwargs",
    "build_point_kwargs",
]


def build_point_kwargs(
    sources,
    marker_size=6,
    palette=None,
    uniform_size=True,
):
    """Build Matplotlib marker keyword arguments for sources.

    Parameters
    ----------
    sources : iterable
        Sources or source labels.
    marker_size : float, optional
        Base marker size. Default is 6.
    palette : list, optional
        Color palette. If None, ``PALETTE_DEFAULT`` is used.
    uniform_size : bool, optional
        Whether to normalize marker sizes according to marker shape.
        Default is True.

    Returns
    -------
    dict
        Mapping from source label to Matplotlib keyword arguments.
    """
    palette = palette or PALETTE_DEFAULT

    result = {}

    for index, source in enumerate(sources):
        label = str(source)
        marker = resolve_marker(label)
        size = resolve_marker_size(marker, marker_size, uniform_size)
        color = palette[index % len(palette)][0]

        result[label] = {
            "label": label,
            "marker": marker,
            "markersize": size,
            "color": color,
        }

    return result


def build_fp_kwargs(
    labels,
    sources=None,
    marker_size=6,
    palette=None,
    uniform_size=True,
):
    """Build plotting keyword arguments for spectral datasets.

    Marker styles are resolved at source level, while colors are assigned
    at dataset level.

    Parameters
    ----------
    labels : iterable of str
        Dataset labels.
    sources : iterable, optional
        Source labels corresponding to each dataset.
    marker_size : float, optional
        Base marker size. Default is 6.
    palette : list, optional
        Color palette. If None, ``PALETTE_DEFAULT`` is used.
    uniform_size : bool, optional
        Whether to normalize marker sizes according to marker shape.
        Default is True.

    Returns
    -------
    dict
        Mapping from source label to plotting keyword arguments.
    """
    palette = palette or PALETTE_DEFAULT

    kwargs_dict = {}

    for index, label in enumerate(labels):
        source_label = str(sources[index]) if sources else label
        marker = resolve_marker(source_label)
        size = resolve_marker_size(marker, marker_size, uniform_size)
        color = palette[index % len(palette)][0]

        kwargs_dict[source_label] = {
            "label": label,
            "marker": marker,
            "markersize": size,
            "color": color,
            "ls": "solid",
        }

    return kwargs_dict
