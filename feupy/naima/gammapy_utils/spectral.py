# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities connecting Naima radiative models with Gammapy models."""

import operator

import naima
from gammapy.modeling.models import (
    CompoundSpectralModel,
    Models,
    NaimaSpectralModel,
    SkyModel,
)

from feupy.naima.particles.analysis import evaluate_particle_spectrum

__all__ = [
    "make_inverse_compton_models",
    "make_pion_decay_models",
    "make_leptohadronic_model",
    "compute_radiative_output",
]


def make_inverse_compton_models(radiative_model, distance):
    """Create Gammapy sky models for inverse-Compton emission.

    Parameters
    ----------
    radiative_model : `naima.radiative.InverseCompton`
        Naima inverse-Compton radiative model.
    distance : `~astropy.units.Quantity`
        Distance to the source.

    Returns
    -------
    models : `~gammapy.modeling.models.Models`
        Sky models for total inverse-Compton emission and each individual
        seed photon field.
    """
    models = Models()

    models.append(
        SkyModel(
            spectral_model=NaimaSpectralModel(
                radiative_model,
                distance=distance,
            ),
            name="IC (total)",
        )
    )

    for seed in radiative_model.seed_photon_fields:
        models.append(
            SkyModel(
                spectral_model=NaimaSpectralModel(
                    radiative_model,
                    seed=seed,
                    distance=distance,
                ),
                name=f"IC ({seed})",
            )
        )

    return models


def make_pion_decay_models(radiative_model, distance):
    """Create a Gammapy sky model for pion-decay emission.

    Parameters
    ----------
    radiative_model : `naima.radiative.PionDecay`
        Naima pion-decay radiative model.
    distance : `~astropy.units.Quantity`
        Distance to the source.

    Returns
    -------
    models : `~gammapy.modeling.models.Models`
        Container with the pion-decay sky model.
    """
    return Models(
        [
            SkyModel(
                spectral_model=NaimaSpectralModel(
                    radiative_model,
                    distance=distance,
                ),
                name="Pion Decay",
            )
        ]
    )


def make_leptohadronic_model(
    ic_models,
    pd_models,
    name="Leptohadronic",
):
    """Create a combined leptohadronic sky model.

    Parameters
    ----------
    ic_models : sequence of `~gammapy.modeling.models.SkyModel`
        Inverse-Compton sky models. The first element must represent the
        total inverse-Compton emission.
    pd_models : sequence of `~gammapy.modeling.models.SkyModel`
        Pion-decay sky models.
    name : str, optional
        Name of the combined sky model.

    Returns
    -------
    model : `~gammapy.modeling.models.SkyModel`
        Combined inverse-Compton plus pion-decay model.

    Raises
    ------
    ValueError
        If ``ic_models`` or ``pd_models`` is empty.
    """
    if not ic_models:
        raise ValueError("ic_models list is empty.")

    if not pd_models:
        raise ValueError("pd_models list is empty.")

    spectral_model = CompoundSpectralModel(
        model1=ic_models[0].spectral_model,
        model2=pd_models[0].spectral_model,
        operator=operator.add,
    )

    return SkyModel(
        spectral_model=spectral_model,
        name=name,
    )


def compute_radiative_output(
    particle_distribution,
    model_type="IC",
    data=None,
    distance=None,
    **kwargs,
):
    """Compute radiative output for IC, PD, or leptohadronic models.

    Parameters
    ----------
    particle_distribution : object or tuple
        Particle distribution. For ``model_type="LH"``, a tuple containing
        the electron and proton distributions is required.
    model_type : {"IC", "PD", "LH"}, optional
        Radiative model type.
    data : `~astropy.units.Quantity`, optional
        Energies at which the radiative flux is evaluated.
    distance : `~astropy.units.Quantity`, optional
        Distance to the source.
    **kwargs : dict
        Parameters required by the selected Naima radiative model.

    Returns
    -------
    result : dict
        Radiative flux, particle-spectrum information, energetics, and
        Gammapy-compatible models.

    Raises
    ------
    ValueError
        If ``model_type`` is not ``"IC"``, ``"PD"``, or ``"LH"``.
    """
    if model_type == "LH":
        particle_dist_e, particle_dist_p = particle_distribution

        eemin = kwargs["Eemin"]
        eemax = kwargs["Eemax"]
        epmin = kwargs["Epmin"]
        epmax = kwargs["Epmax"]

        ic_model = naima.radiative.InverseCompton(
            particle_dist_e,
            seed_photon_fields=kwargs["seed_photon_fields"],
            Eemin=eemin,
            Eemax=eemax,
        )
        pd_model = naima.radiative.PionDecay(
            particle_dist_p,
            nh=kwargs["nh"],
            Epmin=epmin,
            Epmax=epmax,
        )

        return {
            "flux": (
                ic_model.flux(data, distance=distance)
                + pd_model.flux(data, distance=distance)
            ),
            "We": ic_model.compute_We(Eemin=eemin, Eemax=eemax),
            "Wp": pd_model.compute_Wp(Epmin=epmin, Epmax=epmax),
            "particles": {
                "electrons": evaluate_particle_spectrum(
                    ic_model,
                    eemin,
                    eemax,
                ),
                "protons": evaluate_particle_spectrum(
                    pd_model,
                    epmin,
                    epmax,
                ),
            },
            "models": {
                "IC": make_inverse_compton_models(
                    ic_model,
                    distance,
                ),
                "PD": make_pion_decay_models(
                    pd_model,
                    distance,
                ),
            },
        }

    if model_type == "IC":
        emin = kwargs["Eemin"]
        emax = kwargs["Eemax"]

        model = naima.radiative.InverseCompton(
            particle_distribution,
            seed_photon_fields=kwargs["seed_photon_fields"],
            Eemin=emin,
            Eemax=emax,
        )
        energy_content = model.compute_We
        models = make_inverse_compton_models(model, distance)

    elif model_type == "PD":
        emin = kwargs["Epmin"]
        emax = kwargs["Epmax"]

        model = naima.radiative.PionDecay(
            particle_distribution,
            nh=kwargs["nh"],
            Epmin=emin,
            Epmax=emax,
        )
        energy_content = model.compute_Wp
        models = make_pion_decay_models(model, distance)

    else:
        raise ValueError("model_type must be 'IC', 'PD' or 'LH'.")

    return {
        "flux": model.flux(data, distance=distance),
        "W": energy_content(emin, emax),
        "particles": evaluate_particle_spectrum(
            model,
            emin,
            emax,
        ),
        "models": models,
    }
