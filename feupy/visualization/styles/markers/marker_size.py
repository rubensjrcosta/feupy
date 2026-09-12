# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Marker size normalization utilities."""

__all__ = [
    "MARKERS_DEFAULT_DICT",
    "resolve_marker_size",
]


MARKERS_DEFAULT_DICT = {
    "o": (1.0,),
    "s": (1.0,),
    "^": (1.2,),
    "v": (1.2,),
    "*": (1.6,),
    "p": (1.4,),
    "h": (1.3,),
    "8": (1.3,),
    "D": (1.1,),
    ">": (1.2,),
}


def resolve_marker_size(marker, base_size, uniform_size=True):
    """Resolve the marker size used for plotting.

    Parameters
    ----------
    marker : str
        Matplotlib marker symbol.
    base_size : float
        Base marker size.
    uniform_size : bool, optional
        Whether to apply the marker-specific normalization factor.
        Default is True.

    Returns
    -------
    float
        Marker size to use for plotting.
    """
    if not uniform_size:
        return base_size

    scale = MARKERS_DEFAULT_DICT.get(marker, (1.0,))[0]

    return round(base_size * scale, 2)
