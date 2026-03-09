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
    """
    Extract catalog tag from a source label.

    Example
    -------
    >>> extract_catalog_tag("Crab (hgps)")
    'hgps'
    """
    if label is None:
        return None

    match = CATALOG_RE.search(str(label))

    if not match:
        return None

    return match.group(1).lower()


def resolve_marker(label, default="o"):
    """
    Resolve matplotlib marker for a label.

    Parameters
    ----------
    label : str
        Source label containing catalog tag.

    default : str
        Default marker if catalog not recognized.

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