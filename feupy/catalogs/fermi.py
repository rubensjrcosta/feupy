# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for Fermi-LAT 2PC and 3PC catalog sources."""

from gammapy.estimators import FluxPoints

__all__ = [
    "get_flux_points_2PC",
    "get_flux_points_3PC",
]


def get_flux_points_2PC(source):
    """Create flux points for a 2PC catalog source.

    Parameters
    ----------
    source : `~gammapy.catalog.SourceCatalogObject2PC`
        Source from the Fermi-LAT Second Pulsar Catalog (2PC).

    Returns
    -------
    flux_points : `~gammapy.estimators.FluxPoints`
        Flux points built from the source flux-points table and spectral model.
    """
    return FluxPoints.from_table(
        table=source.flux_points_table,
        reference_model=source.spectral_model(),
    )


def get_flux_points_3PC(source, fit="auto"):
    """Create flux points for a 3PC catalog source.

    The 3PC provides spectral fits in which the exponential index is either
    free (``"b free"``) or fixed to 2/3 (``"b 23"``). The ``"auto"`` option
    lets the source select the appropriate fit.

    Parameters
    ----------
    source : `~gammapy.catalog.SourceCatalogObject3PC`
        Source from the Fermi-LAT Third Pulsar Catalog (3PC).
    fit : {"auto", "b free", "b 23"}, optional
        Spectral fit used as the reference model.

    Returns
    -------
    flux_points : `~gammapy.estimators.FluxPoints`
        Flux points built from the source flux-points table and selected
        spectral model.
    """
    return FluxPoints.from_table(
        table=source.flux_points_table,
        reference_model=source.spectral_model(fit),
    )
