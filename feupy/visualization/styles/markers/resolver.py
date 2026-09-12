# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Resolve marker styles from source labels."""

import re

from .registry import CATALOG_STYLE_REGISTRY

__all__ = [
    "extract_catalog_tag",
    "resolve_marker",
]


CATALOG_RE = re.compile(r"\((.*?)\)")


def extract_catalog_tag(label):
    """Extract a catalog tag from a source label.

    Parameters
    ----------
    label : object
        Source label containing a catalog tag in parentheses.

    Returns
    -------
    str or None
        Lowercase catalog tag, or None if no tag is found.

    Examples
    --------
    >>> extract_catalog_tag("Crab (hgps)")
    'hgps'
    """
    if label is None:
        return None

    match = CATALOG_RE.search(str(label))

    if match is None:
        return None

    return match.group(1).lower()


def resolve_marker(label, default="o"):
    """Resolve the Matplotlib marker associated with a source label.

    Parameters
    ----------
    label : object
        Source label containing a catalog tag.
    default : str, optional
        Marker returned when the catalog tag is missing or not registered.
        Default is ``"o"``.

    Returns
    -------
    str
        Matplotlib marker symbol.
    """
    tag = extract_catalog_tag(label)

    if tag is None:
        return default

    style = CATALOG_STYLE_REGISTRY.get(tag)

    if style is None:
        return default

    return style.marker
