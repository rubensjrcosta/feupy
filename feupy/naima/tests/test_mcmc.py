# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.units as u
import naima
import numpy as np
from astropy.table import Table

from feupy.naima.particles.mcmc import (
    _get_amplitude_from_mcmc,
    _get_energy_from_log10,
    _get_mcmc_median,
    make_broken_powerlaw_ep_from_mcmc,
    make_broken_powerlaw_from_mcmc,
    make_exponentialcutoff_powerlaw_from_mcmc,
    make_exponentialcutoffpowerlaw_e_powerlaw_p_from_mcmc,
    make_logparabola_from_mcmc,
    make_powerlaw_from_mcmc,
)


def make_mcmc_table():
    return Table(
        rows=[
            ("log10(norm)", 36.0),
            ("index", 2.1),
            ("alpha", 2.0),
            ("beta", 0.3),
            ("log10(e_break)", 1.0),
            ("index_1", 1.8),
            ("index_2", 3.0),
            ("log10(e_cutoff)", 2.0),
            ("log10(e_break_e)", 1.0),
            ("index_1_e", 1.7),
            ("index_2_e", 2.8),
            ("log10(e_break_p)", 2.0),
            ("index_1_p", 2.0),
            ("index_2_p", 2.6),
            ("index_e", 1.9),
            ("log10(e_cutoff_e)", 2.0),
            ("index_p", 2.2),
        ],
        names=["label", "median"],
    )


def test_get_mcmc_median():
    table = make_mcmc_table()

    assert _get_mcmc_median(table, "index") == 2.1


def test_get_amplitude_from_mcmc():
    table = make_mcmc_table()

    amplitude = _get_amplitude_from_mcmc(table)

    assert amplitude.unit == 1 / u.eV
    assert np.isclose(amplitude.value, 1e36)


def test_get_energy_from_log10():
    table = make_mcmc_table()

    energy = _get_energy_from_log10(table, "log10(e_break)")

    assert energy.unit == u.TeV
    assert np.isclose(energy.value, 10.0)


def test_make_powerlaw_from_mcmc():
    table = make_mcmc_table()

    model = make_powerlaw_from_mcmc(table, e_ref=1 * u.TeV)

    assert isinstance(model, naima.models.PowerLaw)
    assert np.isclose(model.amplitude.value, 1e36)
    assert model.amplitude.unit == 1 / u.eV
    assert model.e_0 == 1 * u.TeV
    assert model.alpha == 2.1


def test_make_logparabola_from_mcmc():
    table = make_mcmc_table()

    model = make_logparabola_from_mcmc(table, e_ref=1 * u.TeV)

    assert isinstance(model, naima.models.LogParabola)
    assert np.isclose(model.amplitude.value, 1e36)
    assert model.e_0 == 1 * u.TeV
    assert model.alpha == 2.0
    assert model.beta == 0.3


def test_make_broken_powerlaw_from_mcmc():
    table = make_mcmc_table()

    model = make_broken_powerlaw_from_mcmc(table, e_ref=1 * u.TeV)

    assert isinstance(model, naima.models.BrokenPowerLaw)
    assert np.isclose(model.amplitude.value, 1e36)
    assert model.e_0 == 1 * u.TeV
    assert model.e_break == 10 * u.TeV
    assert model.alpha_1 == 1.8
    assert model.alpha_2 == 3.0


def test_make_exponentialcutoff_powerlaw_from_mcmc():
    table = make_mcmc_table()

    model = make_exponentialcutoff_powerlaw_from_mcmc(
        table,
        e_ref=1 * u.TeV,
    )

    assert isinstance(model, naima.models.ExponentialCutoffPowerLaw)
    assert np.isclose(model.amplitude.value, 1e36)
    assert model.e_0 == 1 * u.TeV
    assert model.alpha == 2.1
    assert model.e_cutoff == 100 * u.TeV


def test_make_broken_powerlaw_ep_from_mcmc():
    table = make_mcmc_table()

    model_e, model_p = make_broken_powerlaw_ep_from_mcmc(
        table,
        e_ref_e=1 * u.TeV,
        e_ref_p=10 * u.TeV,
        ap_by_ae=100.0,
    )

    assert isinstance(model_e, naima.models.BrokenPowerLaw)
    assert isinstance(model_p, naima.models.BrokenPowerLaw)

    assert np.isclose(model_e.amplitude.value, 1e36)
    assert np.isclose(model_p.amplitude.value, 1e38)

    assert model_e.e_0 == 1 * u.TeV
    assert model_p.e_0 == 10 * u.TeV

    assert model_e.e_break == 10 * u.TeV
    assert model_p.e_break == 100 * u.TeV

    assert model_e.alpha_1 == 1.7
    assert model_e.alpha_2 == 2.8
    assert model_p.alpha_1 == 2.0
    assert model_p.alpha_2 == 2.6


def test_make_exponentialcutoffpowerlaw_e_powerlaw_p_from_mcmc():
    table = make_mcmc_table()

    model_e, model_p = make_exponentialcutoffpowerlaw_e_powerlaw_p_from_mcmc(
        table,
        e_ref_e=1 * u.TeV,
        e_ref_p=10 * u.TeV,
        ap_by_ae=10.0,
    )

    assert isinstance(model_e, naima.models.ExponentialCutoffPowerLaw)
    assert isinstance(model_p, naima.models.PowerLaw)

    assert np.isclose(model_e.amplitude.value, 1e36)
    assert np.isclose(model_p.amplitude.value, 1e37)

    assert model_e.e_0 == 1 * u.TeV
    assert model_p.e_0 == 10 * u.TeV

    assert model_e.alpha == 1.9
    assert model_e.e_cutoff == 100 * u.TeV

    assert model_p.alpha == 2.2
