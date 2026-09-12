# Licensed under a 3-clause BSD style license - see LICENSE.rst

from unittest.mock import MagicMock, patch

import astropy.units as u
import pytest
from gammapy.modeling.models import (
    CompoundSpectralModel,
    Models,
    PowerLawSpectralModel,
    SkyModel,
)

from feupy.naima.gammapy_utils.spectral import (
    compute_radiative_output,
    make_inverse_compton_models,
    make_leptohadronic_model,
    make_pion_decay_models,
)


def test_make_inverse_compton_models():
    radiative_model = MagicMock()
    radiative_model.seed_photon_fields = ["CMB", "FIR"]

    with patch(
        "feupy.naima.gammapy_utils.spectral.NaimaSpectralModel",
        side_effect=lambda *args, **kwargs: PowerLawSpectralModel(),
    ):
        models = make_inverse_compton_models(
            radiative_model,
            1 * u.kpc,
        )

    assert isinstance(models, Models)
    assert len(models) == 3
    assert models.names == ["IC (total)", "IC (CMB)", "IC (FIR)"]


def test_make_pion_decay_models():
    radiative_model = MagicMock()

    with patch(
        "feupy.naima.gammapy_utils.spectral.NaimaSpectralModel",
        return_value=PowerLawSpectralModel(),
    ):
        models = make_pion_decay_models(
            radiative_model,
            1 * u.kpc,
        )

    assert isinstance(models, Models)
    assert len(models) == 1
    assert models.names == ["Pion Decay"]


def test_make_leptohadronic_model():
    ic_model = SkyModel(
        spectral_model=PowerLawSpectralModel(),
        name="IC",
    )
    pd_model = SkyModel(
        spectral_model=PowerLawSpectralModel(),
        name="PD",
    )

    model = make_leptohadronic_model(
        [ic_model],
        [pd_model],
        name="LH",
    )

    assert isinstance(model, SkyModel)
    assert isinstance(
        model.spectral_model,
        CompoundSpectralModel,
    )
    assert model.name == "LH"


def test_make_leptohadronic_model_empty_ic():
    with pytest.raises(ValueError, match="ic_models list is empty"):
        make_leptohadronic_model([], [MagicMock()])


def test_make_leptohadronic_model_empty_pd():
    with pytest.raises(ValueError, match="pd_models list is empty"):
        make_leptohadronic_model([MagicMock()], [])


def test_compute_radiative_output_ic():
    radiative_model = MagicMock()
    radiative_model.flux.return_value = "flux"
    radiative_model.compute_We.return_value = "energy"

    with (
        patch(
            "feupy.naima.gammapy_utils.spectral.naima.radiative.InverseCompton",
            return_value=radiative_model,
        ),
        patch(
            "feupy.naima.gammapy_utils.spectral.evaluate_particle_spectrum",
            return_value="particles",
        ),
        patch(
            "feupy.naima.gammapy_utils.spectral.make_inverse_compton_models",
            return_value="models",
        ),
    ):
        result = compute_radiative_output(
            MagicMock(),
            model_type="IC",
            data="energies",
            distance="distance",
            Eemin=1,
            Eemax=10,
            seed_photon_fields=["CMB"],
        )

    assert result == {
        "flux": "flux",
        "W": "energy",
        "particles": "particles",
        "models": "models",
    }


def test_compute_radiative_output_pd():
    radiative_model = MagicMock()
    radiative_model.flux.return_value = "flux"
    radiative_model.compute_Wp.return_value = "energy"

    with (
        patch(
            "feupy.naima.gammapy_utils.spectral.naima.radiative.PionDecay",
            return_value=radiative_model,
        ),
        patch(
            "feupy.naima.gammapy_utils.spectral.evaluate_particle_spectrum",
            return_value="particles",
        ),
        patch(
            "feupy.naima.gammapy_utils.spectral.make_pion_decay_models",
            return_value="models",
        ),
    ):
        result = compute_radiative_output(
            MagicMock(),
            model_type="PD",
            data="energies",
            distance="distance",
            Epmin=1,
            Epmax=10,
            nh=1,
        )

    assert result == {
        "flux": "flux",
        "W": "energy",
        "particles": "particles",
        "models": "models",
    }


def test_compute_radiative_output_lh():
    ic_model = MagicMock()
    pd_model = MagicMock()

    ic_model.flux.return_value = 1
    pd_model.flux.return_value = 2
    ic_model.compute_We.return_value = "We"
    pd_model.compute_Wp.return_value = "Wp"

    with (
        patch(
            "feupy.naima.gammapy_utils.spectral.naima.radiative.InverseCompton",
            return_value=ic_model,
        ),
        patch(
            "feupy.naima.gammapy_utils.spectral.naima.radiative.PionDecay",
            return_value=pd_model,
        ),
        patch(
            "feupy.naima.gammapy_utils.spectral.evaluate_particle_spectrum",
            side_effect=["electrons", "protons"],
        ),
        patch(
            "feupy.naima.gammapy_utils.spectral.make_inverse_compton_models",
            return_value="ic-models",
        ),
        patch(
            "feupy.naima.gammapy_utils.spectral.make_pion_decay_models",
            return_value="pd-models",
        ),
    ):
        result = compute_radiative_output(
            (MagicMock(), MagicMock()),
            model_type="LH",
            data="energies",
            distance="distance",
            Eemin=1,
            Eemax=10,
            Epmin=2,
            Epmax=20,
            seed_photon_fields=["CMB"],
            nh=1,
        )

    assert result["flux"] == 3
    assert result["We"] == "We"
    assert result["Wp"] == "Wp"
    assert result["particles"] == {
        "electrons": "electrons",
        "protons": "protons",
    }
    assert result["models"] == {
        "IC": "ic-models",
        "PD": "pd-models",
    }


def test_compute_radiative_output_invalid_model_type():
    with pytest.raises(
        ValueError,
        match="model_type must be 'IC', 'PD' or 'LH'",
    ):
        compute_radiative_output(
            MagicMock(),
            model_type="invalid",
        )
