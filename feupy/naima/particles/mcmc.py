# Licensed under a 3-clause BSD style license - see LICENSE.rst
import naima
import astropy.units as u

# ============================================================
# Internal helper functions (not part of public API)
# ============================================================

def _get_mcmc_median(table, label):
    """Return median value for a given MCMC parameter label."""
    return table[table["label"] == label]["median"][0]


def _get_amplitude_from_mcmc(table, label="log10(norm)", unit=u.eV):
    """Return amplitude from log10(norm) parameter."""
    return 10 ** _get_mcmc_median(table, label) / unit


def _get_energy_from_log10(table, label, unit=u.TeV):
    """Return energy from log10(E / unit)."""
    return 10 ** _get_mcmc_median(table, label) * unit


# ============================================================
# Public API – single-population models
# ============================================================

def make_powerlaw_from_mcmc(table, e_ref):
    """Build a Naima PowerLaw model from MCMC results."""
    return naima.models.PowerLaw(
        amplitude=_get_amplitude_from_mcmc(table),
        e_0=e_ref,
        alpha=_get_mcmc_median(table, "index"),
    )


def make_logparabola_from_mcmc(table, e_ref):
    """Build a Naima LogParabola model from MCMC results."""
    return naima.models.LogParabola(
        amplitude=_get_amplitude_from_mcmc(table),
        e_0=e_ref,
        alpha=_get_mcmc_median(table, "alpha"),
        beta=_get_mcmc_median(table, "beta"),
    )


def make_broken_powerlaw_from_mcmc(table, e_ref):
    """Build a Naima BrokenPowerLaw model from MCMC results."""
    return naima.models.BrokenPowerLaw(
        amplitude=_get_amplitude_from_mcmc(table),
        e_0=e_ref,
        e_break=_get_energy_from_log10(table, "log10(e_break/TeV)"),
        alpha_1=_get_mcmc_median(table, "index_1"),
        alpha_2=_get_mcmc_median(table, "index_2"),
    )


def make_exponentialcutoff_powerlaw_from_mcmc(table, e_ref):
    """Build a Naima ExponentialCutoffPowerLaw model from MCMC results."""
    return naima.models.ExponentialCutoffPowerLaw(
        amplitude=_get_amplitude_from_mcmc(table),
        e_0=e_ref,
        alpha=_get_mcmc_median(table, "index"),
        e_cutoff=_get_energy_from_log10(table, "log10(cutoff)"),
    )


# ============================================================
# Public API – electron / proton combined models
# ============================================================

def make_broken_powerlaw_ep_from_mcmc(
    table,
    e_ref_e,
    e_ref_p,
    ap_by_ae=1.0,
):
    """
    Build electron and proton BrokenPowerLaw models from MCMC results.

    Parameters
    ----------
    table : astropy.table.Table
        MCMC summary table.
    e_ref_e, e_ref_p : astropy.units.Quantity
        Reference energies for electrons and protons.
    ap_by_ae : float
        Proton-to-electron normalization ratio.

    Returns
    -------
    model_e, model_p : naima.models.BrokenPowerLaw
    """

    amp_e = _get_amplitude_from_mcmc(table)

    model_e = naima.models.BrokenPowerLaw(
        amplitude=amp_e,
        e_0=e_ref_e,
        e_break=_get_energy_from_log10(table, "log10(e_break_e/TeV)"),
        alpha_1=_get_mcmc_median(table, "index_1_e"),
        alpha_2=_get_mcmc_median(table, "index_2_e"),
    )

    model_p = naima.models.BrokenPowerLaw(
        amplitude=amp_e * ap_by_ae,
        e_0=e_ref_p,
        e_break=_get_energy_from_log10(table, "log10(e_break_p/TeV)"),
        alpha_1=_get_mcmc_median(table, "index_1_p"),
        alpha_2=_get_mcmc_median(table, "index_2_p"),
    )

    return model_e, model_p
