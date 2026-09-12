# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for working with FeuPy catalogs."""

import logging

from .registry import FEUPY_CATALOG_REGISTRY, HAS_FEUPY_DATASETS

log = logging.getLogger(__name__)

if HAS_FEUPY_DATASETS:
    catalog_2fhl = FEUPY_CATALOG_REGISTRY.get_cls("2fhl")()
    catalog_3fhl = FEUPY_CATALOG_REGISTRY.get_cls("3fhl")()
    catalog_3fgl = FEUPY_CATALOG_REGISTRY.get_cls("3fgl")()
    catalog_4fgl = FEUPY_CATALOG_REGISTRY.get_cls("4fgl")()

    catalog_2hwc = FEUPY_CATALOG_REGISTRY.get_cls("2hwc")()
    catalog_3hwc = FEUPY_CATALOG_REGISTRY.get_cls("3hwc")()
    catalog_ehwc = FEUPY_CATALOG_REGISTRY.get_cls("ehwc")()
    catalog_extra_hawc = FEUPY_CATALOG_REGISTRY.get_cls("hwc-2021ApJ")()

    catalog_hgps = FEUPY_CATALOG_REGISTRY.get_cls("hgps")()
    catalog_extra_hess = FEUPY_CATALOG_REGISTRY.get_cls("hess-2019A&A")()

    catalog_gamma_cat = FEUPY_CATALOG_REGISTRY.get_cls("gamma-cat")()

    catalog_vtscat = FEUPY_CATALOG_REGISTRY.get_cls("vtscat")()
    catalog_veritas = FEUPY_CATALOG_REGISTRY.get_cls("veritas-2018ApJ")()

    catalog_lhaaso = FEUPY_CATALOG_REGISTRY.get_cls("LHAASO")()
    catalog_1lhaaso = FEUPY_CATALOG_REGISTRY.get_cls("1LHAASO")()
    catalog_extra_lhaaso = FEUPY_CATALOG_REGISTRY.get_cls("LHAASO-2024icrc")()

    catalog_psrcat = FEUPY_CATALOG_REGISTRY.get_cls("psrcat")()


def load_catalogs(catalogs=None):
    """Load catalog instances from a registry.

    Parameters
    ----------
    catalogs : `~gammapy.utils.registry.Registry`, optional
        Catalog registry to load. If None, ``FEUPY_CATALOG_REGISTRY`` is used.

    Returns
    -------
    source_catalogs : list
        Instantiated source catalogs.

    Raises
    ------
    ValueError
        If a catalog cannot be instantiated.
    """
    if catalogs is None:
        catalogs = FEUPY_CATALOG_REGISTRY

    source_catalogs = []

    for index, catalog in enumerate(catalogs):
        try:
            catalog_instance = catalogs.get_cls(catalog.tag)()
        except Exception as error:
            log.error(
                "Failed to load catalog '%s' at index %d: %s",
                catalog.tag,
                index,
                error,
            )
            raise ValueError(
                f"Error loading catalog '{catalog.tag}' at index {index}: {error}"
            ) from error

        source_catalogs.append(catalog_instance)

    log.info("Loaded %d catalogs.", len(source_catalogs))
    return source_catalogs


def get_catalog_tag(source):
    """Return the catalog tag associated with a source object.

    Parameters
    ----------
    source : object
        Source object whose catalog tag should be identified.

    Returns
    -------
    tag : str
        Tag of the matching catalog.

    Raises
    ------
    ValueError
        If no catalog in ``FEUPY_CATALOG_REGISTRY`` matches the source.
    """
    matching_catalog = next(
        (
            catalog
            for catalog in FEUPY_CATALOG_REGISTRY
            if isinstance(source, catalog.source_object_class)
        ),
        None,
    )

    if matching_catalog is None:
        log.error("Failed to find catalog for source: %s", source)
        raise ValueError(f"No matching catalog found for source: {source}")

    return matching_catalog.tag
