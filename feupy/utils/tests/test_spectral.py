# Licensed under a 3-clause BSD style license - see LICENSE

import astropy.units as u
import pytest
from gammapy.modeling.models import (
    ExpCutoffPowerLawSpectralModel,
    PowerLawSpectralModel,
    SkyModel,
)

from feupy.utils import spectral
from feupy.utils.spectral import get_ecut_from_ecpl


def make_ecpl_model(lambda_error=0.01):
    """Create an ECPL sky model for testing."""
    spectral_model = ExpCutoffPowerLawSpectralModel(
        amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
        index=2.0,
        lambda_=0.1 / u.TeV,
        reference=1 * u.TeV,
    )
    spectral_model.lambda_.error = lambda_error

    return SkyModel(spectral_model=spectral_model)


def test_all():
    assert spectral.__all__ == ["get_ecut_from_ecpl"]
    assert hasattr(spectral, "get_ecut_from_ecpl")


def test_get_ecut_from_ecpl():
    model = make_ecpl_model()

    result = get_ecut_from_ecpl(model)

    assert result == r"10.00 \pm 1.00"


def test_get_ecut_from_ecpl_custom_format():
    model = make_ecpl_model()

    result = get_ecut_from_ecpl(model, fmt="{:.1f} +/- {:.1f}")

    assert result == "10.0 +/- 1.0"


def test_get_ecut_from_ecpl_wrong_model():
    spectral_model = PowerLawSpectralModel(
        amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
        index=2.0,
        reference=1 * u.TeV,
    )
    model = SkyModel(spectral_model=spectral_model)

    with pytest.raises(TypeError, match="lambda_"):
        get_ecut_from_ecpl(model)


@pytest.mark.parametrize("lambda_error", [0, -0.01])
def test_get_ecut_from_ecpl_invalid_error(lambda_error):
    model = make_ecpl_model(lambda_error=lambda_error)

    with pytest.raises(ValueError, match="no associated error"):
        get_ecut_from_ecpl(model)
