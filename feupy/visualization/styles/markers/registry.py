# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Catalog style registry."""

__all__ = [
    "CATALOG_STYLE_REGISTRY",
    "CatalogStyle",
    "CatalogStyleRegistry",
]


class CatalogStyle:
    """Metadata describing the plotting style of a catalog.

    Parameters
    ----------
    tag : str
        Catalog tag, for example ``"3fgl"`` or ``"4fgl"``.
    marker : str, optional
        Matplotlib marker symbol. Default is ``"o"``.
    color : str, optional
        Default color associated with the catalog.
    """

    def __init__(self, tag, marker="o", color=None):
        self.tag = tag.lower()
        self.marker = marker
        self.color = color


class CatalogStyleRegistry:
    """Registry storing plotting styles for catalogs."""

    def __init__(self):
        self._registry = {}

    def register(self, style):
        """Register a catalog style.

        Parameters
        ----------
        style : `CatalogStyle`
            Catalog style to register.
        """
        self._registry[style.tag] = style

    def get(self, tag):
        """Retrieve the style associated with a catalog tag.

        Parameters
        ----------
        tag : str
            Catalog tag.

        Returns
        -------
        `CatalogStyle` or None
            Registered catalog style, or None if the tag is unknown.
        """
        return self._registry.get(tag.lower())

    def tags(self):
        """Return the registered catalog tags.

        Returns
        -------
        list of str
            Registered catalog tags.
        """
        return list(self._registry.keys())


CATALOG_STYLE_REGISTRY = CatalogStyleRegistry()
