# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities to evaluate particle energy distributions."""

import astropy.units as u
import numpy as np

__all__ = ["evaluate_particle_spectrum"]


def evaluate_particle_spectrum(
    radiative_model,
    e_min,
    e_max,
    energy=None,
    n_points=100,
):
    """Evaluate a particle spectrum and its energy-weighted form.

    Parameters
    ----------
    radiative_model : object
        Radiative model providing a ``particle_distribution`` callable.
    e_min : `~astropy.units.Quantity`
        Minimum particle energy.
    e_max : `~astropy.units.Quantity`
        Maximum particle energy.
    energy : `~astropy.units.Quantity`, optional
        Energy grid. If None, a logarithmically spaced grid between
        ``e_min`` and ``e_max`` is created.
    n_points : int, optional
        Number of points in the generated energy grid.

    Returns
    -------
    energy : `~astropy.units.Quantity`
        Particle energy grid.
    dnde : `~astropy.units.Quantity`
        Differential particle distribution.
    e2dnde : `~astropy.units.Quantity`
        Energy-weighted particle distribution, :math:`E^2 dN/dE`.
    """
    if energy is None:
        energy = (
            np.logspace(
                np.log10(e_min.to_value(u.TeV)),
                np.log10(e_max.to_value(u.TeV)),
                n_points,
            )
            * u.TeV
        )

    dnde = radiative_model.particle_distribution(energy)
    e2dnde = (energy**2 * dnde).to("erg")

    return energy, dnde, e2dnde
