# Licensed under a 3-clause BSD style license - see LICENSE.rst

import naima
from gammapy.modeling.models import NaimaSpectralModel, CompoundSpectralModel, Models, SkyModel
import operator
from feupy.naima.particles.analysis import evaluate_particle_spectrum

__all__ = [
    "make_inverse_compton_models",
    "make_pion_decay_models",
    "make_leptohadronic_model",
    "compute_radiative_output"
]

def make_inverse_compton_models(radiative_model, distance):
    """Create SkyModels for inverse Compton emission.

    Parameters
    ----------
    radiative_model : naima.radiative.RadiativeModel
        Naima radiative model instance.
    distance : `~astropy.units.Quantity`
        Distance to the source.

    Returns
    -------
    models : `~gammapy.modeling.models.Models`
        SkyModels for total IC emission and individual seed photon fields.
    """
    models = Models()

    models.append(
        SkyModel(
            spectral_model=NaimaSpectralModel(
                radiative_model, distance=distance
            ),
            name="IC (total)",
        )
    )

    for seed in radiative_model.seed_photon_fields:
        models.append(
            SkyModel(
                spectral_model=NaimaSpectralModel(
                    radiative_model, seed=seed, distance=distance
                ),
                name=f"IC ({seed})",
            )
        )

    return models


def make_pion_decay_models(radiative_model, distance):
    """Create SkyModels for Pion Decay emission.
    
    Parameters
    ----------
    radiative_model : naima.radiative.RadiativeModel
        Naima radiative model instance.
    distance : `~astropy.units.Quantity`
        Distance to the source.
    
    Returns
    -------
    models : `~gammapy.modeling.models.Models`
        SkyModels for Pion Decay emission.
    """
    models = Models()

    models.append(
        SkyModel(
            spectral_model=NaimaSpectralModel(
                radiative_model, distance=distance
            ),
            name="Pion Decay",
        )
    )
    return models

def make_leptohadronic_model(
    ic_models,
    pd_models,
    name: str = "Leptohadronic",
):
    """
    Create a combined leptohadronic SkyModel (IC + pion decay).

    Parameters
    ----------
    ic_models : list of `~gammapy.modeling.models.SkyModel`
        Inverse Compton SkyModels. The first element must correspond
        to the total IC emission.
    pd_models : list of `~gammapy.modeling.models.SkyModel`
        Pion decay SkyModels (typically a single-element list).
    name : str, optional
        Name of the combined SkyModel.

    Returns
    -------
    model : `~gammapy.modeling.models.SkyModel`
        Combined leptohadronic SkyModel.
    """
    Models()
    
    if not ic_models:
        raise ValueError("ic_models list is empty.")

    if not pd_models:
        raise ValueError("pd_models list is empty.")

    spec_ic = ic_models[0].spectral_model
    spec_pd = pd_models[0].spectral_model

    spectral_model = CompoundSpectralModel(
        model1=spec_ic,
        model2=spec_pd,
        operator=operator.add,
    )

    return SkyModel(spectral_model=spectral_model, name=name)

def compute_radiative_output(
    particle_distribution,
    model_type="IC",
    data=None,
    distance=None,
    **kwargs,
):
    """
    Compute radiative output for IC, PD or Lepto-Hadronic models.

    Parameters
    ----------
    particle_distribution : naima.models.ParticleDistribution or tuple
        Electron distribution (IC, PD) or (electrons, protons) for LH.
    model_type : {"IC", "PD", "LH"}
        Radiative model type.
    data : `~astropy.units.Quantity`
        Energy array where flux is evaluated.
    distance : `~astropy.units.Quantity`
        Source distance.

    Returns
    -------
    result : dict
        Dictionary containing flux, particle distributions, energetics
        and Gammapy-compatible models.
    """
    result = {}

    # ============================
    # Lepto-Hadronic
    # ============================
    if model_type == "LH":
        particle_dist_e, particle_dist_p = particle_distribution

        Eemin = kwargs["Eemin"]
        Eemax = kwargs["Eemax"]
        Epmin = kwargs["Epmin"]
        Epmax = kwargs["Epmax"]

        IC = naima.radiative.InverseCompton(
            particle_dist_e,
            seed_photon_fields=kwargs["seed_photon_fields"],
            Eemin=Eemin,
            Eemax=Eemax,
        )

        PD = naima.radiative.PionDecay(
            particle_dist_p,
            nh=kwargs["nh"],
            Epmin=Epmin,
            Epmax=Epmax,
        )

        result["flux"] = (
            IC.flux(data, distance=distance)
            + PD.flux(data, distance=distance)
        )

        result["We"] = IC.compute_We(Eemin=Eemin, Eemax=Eemax)
        result["Wp"] = PD.compute_Wp(Epmin=Epmin, Epmax=Epmax)

        result["particles"] = {
            "electrons": evaluate_particle_spectrum(
                IC, Eemin, Eemax
            ),
            "protons": evaluate_particle_spectrum(
                PD, Epmin, Epmax
            ),
        }

        result["models"] = {
            "IC": make_inverse_compton_models(IC, distance),
            "PD": make_pion_decay_models(PD, distance),
        }

        return result

    # ============================
    # Pure IC or PD
    # ============================
    if model_type == "IC":
        Emin, Emax = kwargs["Eemin"], kwargs["Eemax"]

        model = naima.radiative.InverseCompton(
            particle_distribution,
            seed_photon_fields=kwargs["seed_photon_fields"],
            Eemin=Emin,
            Eemax=Emax,
        )

        energy_content = model.compute_We
        models = make_inverse_compton_models(model, distance)

    elif model_type == "PD":
        Emin, Emax = kwargs["Epmin"], kwargs["Epmax"]

        model = naima.radiative.PionDecay(
            particle_distribution,
            nh=kwargs["nh"],
            Epmin=Emin,
            Epmax=Emax,
        )

        energy_content = model.compute_Wp
        models = make_pion_decay_models(model, distance)

    else:
        raise ValueError("model_type must be 'IC', 'PD' or 'LH'.")

    result["flux"] = model.flux(data, distance=distance)
    result["W"] = energy_content(Emin, Emax)

    result["particles"] = evaluate_particle_spectrum(
        model, Emin, Emax
    )

    result["models"] = models

    return result
