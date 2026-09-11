# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Catalog style registry."""

__all__ = [
    "CatalogStyle",
    "CatalogStyleRegistry",
    "CATALOG_STYLE_REGISTRY",
]


class CatalogStyle:
    """
    Metadata describing plotting style for a catalog.

    Parameters
    ----------
    tag : str
        Catalog tag (e.g. ``"3fgl"``, ``"4fgl"``).
    marker : str, optional
        Matplotlib marker symbol.
    color : str, optional
        Default color associated with this catalog.
    """

    def __init__(self, tag, marker="o", color=None):
        self.tag = tag.lower()
        self.marker = marker
        self.color = color


class CatalogStyleRegistry:
    """
    Registry storing plotting styles for catalogs.
    """

    def __init__(self):
        self._registry = {}

    def register(self, style):
        """
        Register a catalog style.
        """
        self._registry[style.tag] = style

    def get(self, tag):
        """
        Retrieve style for a catalog tag.
        """
        return self._registry.get(tag.lower())

    def tags(self):
        """
        Return list of registered catalog tags.
        """
        return list(self._registry.keys())


# Global registry instance
CATALOG_STYLE_REGISTRY = CatalogStyleRegistry()
