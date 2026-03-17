# Licensed under a 3-clause BSD style license - see LICENSE.rst"

import logging
import astropy.units as u
import numpy as np

__all__ = ["evaluate_particle_spectrum"]

log = logging.getLogger(__name__)

def evaluate_particle_spectrum(
    radiative_model,
    E_min,
    E_max,
    energy=None,
    n_points=100,
):
    """
    Compute the particle energy distribution and its E^2-weighted form.

    Parameters
    ----------
    radiative_model : naima.radiative.BaseRadiativeModel
        Radiative model providing the particle distribution.
    E_min, E_max : `~astropy.units.Quantity`
        Minimum and maximum particle energies.
    energy : `~astropy.units.Quantity`, optional
        Energy grid. If None, a log-spaced grid is created.
    n_points : int, optional
        Number of energy points if energy grid is not provided.

    Returns
    -------
    energy : `~astropy.units.Quantity`
        Particle energy grid.
    dnde : `~astropy.units.Quantity`
        Differential particle distribution dN/dE.
    e2dnde : `~astropy.units.Quantity`
        Energy-weighted particle distribution E^2 dN/dE.
    """
    if energy is None:
        energy = np.logspace(
            np.log10(E_min.to_value("TeV")),
            np.log10(E_max.to_value("TeV")),
            n_points,
        ) * u.TeV

    dnde = radiative_model.particle_distribution(energy)

    e2dnde = (energy**2 * dnde).to("erg")

    return energy, dnde, e2dnde
