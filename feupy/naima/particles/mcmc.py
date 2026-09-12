# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities to build Naima particle distributions from MCMC results."""

import astropy.units as u
import naima

__all__ = [
    "make_powerlaw_from_mcmc",
    "make_logparabola_from_mcmc",
    "make_broken_powerlaw_from_mcmc",
    "make_exponentialcutoff_powerlaw_from_mcmc",
    "make_broken_powerlaw_ep_from_mcmc",
    "make_exponentialcutoffpowerlaw_e_powerlaw_p_from_mcmc",
]


def _get_mcmc_median(table, label):
    """Return the median value of an MCMC parameter."""
    return table[table["label"] == label]["median"][0]


def _get_amplitude_from_mcmc(table, label="log10(norm)", unit=u.eV):
    """Return particle normalization from a logarithmic MCMC parameter."""
    return 10 ** _get_mcmc_median(table, label) / unit


def _get_energy_from_log10(table, label, unit=u.TeV):
    """Return energy from a logarithmic MCMC parameter."""
    return 10 ** _get_mcmc_median(table, label) * unit


def make_powerlaw_from_mcmc(table, e_ref):
    """Create a Naima power-law model from MCMC results.

    Parameters
    ----------
    table : `~astropy.table.Table`
        MCMC summary table containing ``label`` and ``median`` columns.
    e_ref : `~astropy.units.Quantity`
        Reference energy of the particle distribution.

    Returns
    -------
    model : `naima.models.PowerLaw`
        Power-law particle distribution.
    """
    return naima.models.PowerLaw(
        amplitude=_get_amplitude_from_mcmc(table),
        e_0=e_ref,
        alpha=_get_mcmc_median(table, "index"),
    )


def make_logparabola_from_mcmc(table, e_ref):
    """Create a Naima log-parabola model from MCMC results.

    Parameters
    ----------
    table : `~astropy.table.Table`
        MCMC summary table containing ``label`` and ``median`` columns.
    e_ref : `~astropy.units.Quantity`
        Reference energy of the particle distribution.

    Returns
    -------
    model : `naima.models.LogParabola`
        Log-parabola particle distribution.
    """
    return naima.models.LogParabola(
        amplitude=_get_amplitude_from_mcmc(table),
        e_0=e_ref,
        alpha=_get_mcmc_median(table, "alpha"),
        beta=_get_mcmc_median(table, "beta"),
    )


def make_broken_powerlaw_from_mcmc(table, e_ref):
    """Create a Naima broken power-law model from MCMC results.

    Parameters
    ----------
    table : `~astropy.table.Table`
        MCMC summary table containing ``label`` and ``median`` columns.
    e_ref : `~astropy.units.Quantity`
        Reference energy of the particle distribution.

    Returns
    -------
    model : `naima.models.BrokenPowerLaw`
        Broken power-law particle distribution.
    """
    return naima.models.BrokenPowerLaw(
        amplitude=_get_amplitude_from_mcmc(table),
        e_0=e_ref,
        e_break=_get_energy_from_log10(table, "log10(e_break)"),
        alpha_1=_get_mcmc_median(table, "index_1"),
        alpha_2=_get_mcmc_median(table, "index_2"),
    )


def make_exponentialcutoff_powerlaw_from_mcmc(table, e_ref):
    """Create a Naima exponential-cutoff power-law model from MCMC results.

    Parameters
    ----------
    table : `~astropy.table.Table`
        MCMC summary table containing ``label`` and ``median`` columns.
    e_ref : `~astropy.units.Quantity`
        Reference energy of the particle distribution.

    Returns
    -------
    model : `naima.models.ExponentialCutoffPowerLaw`
        Exponential-cutoff power-law particle distribution.
    """
    return naima.models.ExponentialCutoffPowerLaw(
        amplitude=_get_amplitude_from_mcmc(table),
        e_0=e_ref,
        alpha=_get_mcmc_median(table, "index"),
        e_cutoff=_get_energy_from_log10(table, "log10(e_cutoff)"),
    )


def make_broken_powerlaw_ep_from_mcmc(
    table,
    e_ref_e,
    e_ref_p,
    ap_by_ae=1.0,
):
    """Create electron and proton broken power-law models from MCMC results.

    Parameters
    ----------
    table : `~astropy.table.Table`
        MCMC summary table containing ``label`` and ``median`` columns.
    e_ref_e : `~astropy.units.Quantity`
        Electron reference energy.
    e_ref_p : `~astropy.units.Quantity`
        Proton reference energy.
    ap_by_ae : float, optional
        Proton-to-electron normalization ratio.

    Returns
    -------
    model_e : `naima.models.BrokenPowerLaw`
        Electron particle distribution.
    model_p : `naima.models.BrokenPowerLaw`
        Proton particle distribution.
    """
    amplitude_e = _get_amplitude_from_mcmc(table)

    model_e = naima.models.BrokenPowerLaw(
        amplitude=amplitude_e,
        e_0=e_ref_e,
        e_break=_get_energy_from_log10(table, "log10(e_break_e)"),
        alpha_1=_get_mcmc_median(table, "index_1_e"),
        alpha_2=_get_mcmc_median(table, "index_2_e"),
    )

    model_p = naima.models.BrokenPowerLaw(
        amplitude=amplitude_e * ap_by_ae,
        e_0=e_ref_p,
        e_break=_get_energy_from_log10(table, "log10(e_break_p)"),
        alpha_1=_get_mcmc_median(table, "index_1_p"),
        alpha_2=_get_mcmc_median(table, "index_2_p"),
    )

    return model_e, model_p


def make_exponentialcutoffpowerlaw_e_powerlaw_p_from_mcmc(
    table,
    e_ref_e,
    e_ref_p,
    ap_by_ae=1.0,
):
    """Create electron cutoff and proton power-law models from MCMC results.

    Parameters
    ----------
    table : `~astropy.table.Table`
        MCMC summary table containing ``label`` and ``median`` columns.
    e_ref_e : `~astropy.units.Quantity`
        Electron reference energy.
    e_ref_p : `~astropy.units.Quantity`
        Proton reference energy.
    ap_by_ae : float, optional
        Proton-to-electron normalization ratio.

    Returns
    -------
    model_e : `naima.models.ExponentialCutoffPowerLaw`
        Electron particle distribution.
    model_p : `naima.models.PowerLaw`
        Proton particle distribution.
    """
    amplitude_e = _get_amplitude_from_mcmc(table)

    model_e = naima.models.ExponentialCutoffPowerLaw(
        amplitude=amplitude_e,
        e_0=e_ref_e,
        alpha=_get_mcmc_median(table, "index_e"),
        e_cutoff=_get_energy_from_log10(table, "log10(e_cutoff_e)"),
    )

    model_p = naima.models.PowerLaw(
        amplitude=amplitude_e * ap_by_ae,
        e_0=e_ref_p,
        alpha=_get_mcmc_median(table, "index_p"),
    )

    return model_e, model_p
