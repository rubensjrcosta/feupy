# Licensed under a 3-clause BSD style license - see LICENSE.rst

import astropy.units as u
import numpy as np

from feupy.naima.particles.analysis import evaluate_particle_spectrum


class DummyRadiativeModel:
    """Minimal radiative model for testing."""

    @staticmethod
    def particle_distribution(energy):
        return 1e36 * (energy / (1 * u.TeV)) ** -2 / u.TeV


def test_evaluate_particle_spectrum_generated_grid():
    model = DummyRadiativeModel()

    energy, dnde, e2dnde = evaluate_particle_spectrum(
        model,
        e_min=1 * u.TeV,
        e_max=100 * u.TeV,
        n_points=3,
    )

    assert len(energy) == 3
    assert np.allclose(
        energy.to_value(u.TeV),
        [1.0, 10.0, 100.0],
    )
    assert dnde.unit.is_equivalent(1 / u.TeV)
    assert e2dnde.unit == u.erg


def test_evaluate_particle_spectrum_custom_grid():
    model = DummyRadiativeModel()
    input_energy = [1.0, 2.0, 4.0] * u.TeV

    energy, dnde, e2dnde = evaluate_particle_spectrum(
        model,
        e_min=1 * u.TeV,
        e_max=10 * u.TeV,
        energy=input_energy,
    )

    assert energy is input_energy
    assert len(dnde) == 3
    assert len(e2dnde) == 3


def test_evaluate_particle_spectrum_values():
    model = DummyRadiativeModel()
    input_energy = [1.0, 10.0] * u.TeV

    energy, dnde, e2dnde = evaluate_particle_spectrum(
        model,
        e_min=1 * u.TeV,
        e_max=10 * u.TeV,
        energy=input_energy,
    )

    assert np.allclose(
        dnde.to_value(1 / u.TeV),
        [1e36, 1e34],
    )

    expected = (energy**2 * dnde).to(u.erg)
    assert np.allclose(
        e2dnde.to_value(u.erg),
        expected.to_value(u.erg),
    )
