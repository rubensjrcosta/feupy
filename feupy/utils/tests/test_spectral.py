# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import astropy.units as u

from gammapy.modeling.models import (
    SkyModel,
    ExpCutoffPowerLawSpectralModel,
    PowerLawSpectralModel,
)

from feupy.utils.spectral import get_ecut_from_ecpl


def test_get_ecut_from_ecpl():
    spectral_model = ExpCutoffPowerLawSpectralModel(
        amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
        index=2.0,
        lambda_=0.1 / u.TeV,
        reference=1 * u.TeV,
    )

    # define error manually
    spectral_model.lambda_.error = 0.01

    model = SkyModel(spectral_model=spectral_model)

    result = get_ecut_from_ecpl(model)

    ecut = 1 / 0.1
    ecut_err = ecut * (0.01 / 0.1)

    expected = f"{ecut:.2f} \\pm {ecut_err:.2f}"

    assert result == expected


def test_get_ecut_from_ecpl_wrong_model():
    spectral_model = PowerLawSpectralModel(
        amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
        index=2.0,
        reference=1 * u.TeV,
    )

    model = SkyModel(spectral_model=spectral_model)

    with pytest.raises(TypeError):
        get_ecut_from_ecpl(model)


def test_get_ecut_from_ecpl_no_error():
    spectral_model = ExpCutoffPowerLawSpectralModel(
        amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
        index=2.0,
        lambda_=0.1 / u.TeV,
        reference=1 * u.TeV,
    )

    # Explicitly set zero error
    spectral_model.lambda_.error = 0

    model = SkyModel(spectral_model=spectral_model)

    with pytest.raises(ValueError):
        get_ecut_from_ecpl(model)