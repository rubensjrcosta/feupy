# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Naima."""

import numpy as np

from astropy.table import Table, Column
from astropy import units as u
from astropy.units import Quantity

from gammapy.estimators.map.core import DEFAULT_UNIT, OPTIONAL_QUANTITIES,REQUIRED_COLUMNS

from naima.plot import find_ML
from gammapy.modeling.models import NaimaSpectralModel, Models, SkyModel

# def get_radiative_model_IC_PL(
#     pars, 
#     **kwargs
# ):
#     """
#     """
#     kwargs.setdefault('ee_0', 1*u.TeV)
#     ee_0 = kwargs['ee_0']
    
#     kwargs.setdefault('Eemin', 1*u.GeV)
#     Eemin = kwargs['Eemin']
    
#     kwargs.setdefault('Eemax', 510*u.TeV)
#     Eemax = kwargs['Eemax']
    
    
#     kwargs.setdefault('seed_photon_fields', ["CMB", 2.7 * u.K, 0.25 * u.eV / u.cm**3])
#     seed_photon_fields = kwargs['seed_photon_fields']
    
#     amplitude = 10 ** pars[0] / u.eV
#     alpha = pars[1]
    
#     PARTICLE_DISTRIBUTION = naima.models.PowerLaw(
#         amplitude, 
#         ee_0,  
#         alpha
#     )

#     radiative_model = naima.radiative.InverseCompton(
#         PARTICLE_DISTRIBUTION,
#         seed_photon_fields=seed_photon_fields,
#         Eemin=Eemin,
#         Eemax=Eemax,
#     )
    
    
#     return radiative_model
def get_inverse_compton_models(radiative_model, distance):
    models = Models()

    seeds = list(radiative_model.seed_photon_fields.keys())
    spectral_model = NaimaSpectralModel(radiative_model, distance=distance)
    model = SkyModel(spectral_model=spectral_model, name="IC (total)")
    models.append(model)
    
    for index, seed in enumerate(seeds):
        spectral_model = NaimaSpectralModel(radiative_model, seed=seed, distance=distance)
        model = SkyModel(spectral_model=spectral_model, name=f"IC ({seed})")
        models.append(model)
    return models

def calc_BIC(sampler):
    """Compute the Bayesian Information Criterion (BIC).
    
    Parameters
    ----------
    sampler :`~emcee.EnsembleSampler`
        Ensemble sampler with walker positions after ``nburn`` burn-in steps.
            
    Returns
    -------
    BIC
    
    """   
        
    MLp = find_ML(sampler, modelidx=0)[1]
    ML = find_ML(sampler, modelidx=0)[0]
    
    return len(MLp) * np.log(len(sampler.data)) - 2 * ML


REQUIRED_NAIMA_COLUMNS_NAMES = {
    'e_ref': 'energy',
    'e_min': 'energy_error_lo',
    'e_max': 'energy_error_hi',
    'dnde': 'flux',
    'dnde_err': 'flux_error',
    'dnde_errp': 'flux_error_hi',
    'dnde_errn': 'flux_error_lo',
    'dnde_ul': 'flux_ul',
    'e2dnde': 'flux',
    'e2dnde_err': 'flux_error',
    'e2dnde_errp': 'flux_error_hi',
    'e2dnde_errn': 'flux_error_lo',
    'e2dnde_ul': 'flux_ul',
    'is_ul': 'ul',
}

REQUIRED_NAIMA_COLUMNS = {
    'dnde':   ['e_ref', 'dnde', 'dnde_err', 'dnde_errp', 'dnde_errn', 'dnde_ul', 'is_ul'],
    'e2dnde':   ['e_ref', 'e2dnde', 'e2dnde_err', 'e2dnde_errp', 'e2dnde_errn', 'e2dnde_ul', 'is_ul'],
}   
    
def make_naima_tables(datasets, sed_type="dnde"):
    tables = []
    
    for index, dataset in enumerate(datasets):
        table = Table()
        table.meta['name'] = f'{dataset.name}'
#         table.meta['keywords']['cl']=0.99
    
        data = dataset.data.to_table(sed_type=sed_type)
        colnames = data.colnames
        columns = [x for x in [_ if _ in colnames else None for _ in REQUIRED_NAIMA_COLUMNS[sed_type]] if x is not None]
        columns_naima = [REQUIRED_NAIMA_COLUMNS_NAMES[x] for x in [_ if _ in colnames else None for _ in REQUIRED_NAIMA_COLUMNS[sed_type]] if x is not None]
        
    
        for column, column_naima in zip(columns, columns_naima):
            table[column_naima] = data[column]


        tables.append(table)
    return tables

def compute_particle_distribution(radiative_model, E_min, E_max):
    """
    Compute the particle distribution and its energy-weighted version.

    This function calculates the particle energy distribution based on a given radiative model 
    and returns the energy array, the particle distribution function values, 
    and the energy-weighted distribution in units of erg.

    Parameters
    ----------
    radiative_model : object
        A model providing the particle distribution function. 
        Must have a method `particle_distribution(energy)`.
    E_min : `~astropy.units.Quantity`
        Minimum energy for the particle distribution (e.g., in TeV).
    E_max : `~astropy.units.Quantity`
        Maximum energy for the particle distribution (e.g., in TeV).

    Returns
    -------
    energy : `~astropy.units.Quantity`
        Array of particle energies (e.g., in TeV).
    function_energy : `~astropy.units.Quantity`
        Particle distribution function evaluated at the given energies.
    energy_weighted : `~astropy.units.Quantity`
        Energy-weighted particle distribution, (energy^2 * function_energy), in erg.
    """
    energy = np.logspace(
        np.log10(E_min.to("TeV").value), 
        np.log10(E_max.to("TeV").value), 
        100
    ) * u.TeV
    function_energy = radiative_model.particle_distribution(energy)
    energy_weighted = (energy**2 * function_energy).to("erg")

    return energy, function_energy, energy_weighted


# def particle_distribution(energy, function_energy):
#     """Particle's distibution.

#     Parameters
#     ----------
#     energy : `~astropy.units.Quantity`
#         Particle energy
#     function_energy : 
#         Particle distribution function
    
#     Returns
#     -------
#     energy, (energy**2 * function_energy).to('erg')
    
#     """   
#     return energy, (energy**2 * function_energy).to('erg')


def get_sed_e2dnde(model_func, photon_energy, sed_type='e2dnde'):
        return (model_func*photon_energy)*photon_energy    
    

## Additional model expressions

import astropy.units as u
import numpy as np

import naima

################################################################################
#
# This file shows a few example model functions (with associated priors, labels
# and p0 vector), that can be used as input for naima.run_sampler
#
################################################################################

#
# RADIATIVE MODELS
#
# Pion decay
# ==========

PionDecay_ECPL_p0 = np.array((46, 2.34, np.log10(80.0)))
PionDecay_ECPL_labels = ["log10(norm)", "index", "log10(cutoff)"]

# Prepare an energy array for saving the particle distribution
proton_energy = np.logspace(-3, 2, 50) * u.TeV


def PionDecay_ECPL(pars, data):
    amplitude = 10 ** pars[0] / u.TeV
    alpha = pars[1]
    e_cutoff = 10 ** pars[2] * u.TeV

    ECPL = naima.models.ExponentialCutoffPowerLaw(
        amplitude, 30 * u.TeV, alpha, e_cutoff
    )
    PP = naima.models.PionDecay(ECPL, nh=1.0 * u.cm ** -3)

    model = PP.flux(data, distance=1.0 * u.kpc)
    # Save a realization of the particle distribution to the metadata blob
    proton_dist = PP.particle_distribution(proton_energy)
    # Compute the total energy in protons above 1 TeV for this realization
    Wp = PP.compute_Wp(Epmin=1 * u.TeV)

    # Return the model, proton distribution and energy in protons to be stored
    # in metadata blobs
    return model, (proton_energy, proton_dist), Wp


def PionDecay_ECPL_lnprior(pars):
    logprob = naima.uniform_prior(pars[1], -1, 5)
    return logprob


# Inverse Compton with the energy in electrons as the normalization parameter
# ===========================================================================

IC_We_p0 = np.array((40, 3.0, np.log10(30)))
IC_We_labels = ["log10(We)", "index", "log10(cutoff)"]


def IC_We(pars, data):
    # Example of a model that is normalized though the total energy in electrons

    # Match parameters to ECPL properties, and give them the appropriate units
    We = 10 ** pars[0] * u.erg
    alpha = pars[1]
    e_cutoff = 10 ** pars[2] * u.TeV

    # Initialize instances of the particle distribution and radiative model
    # set a bogus normalization that will be changed in third line
    ECPL = naima.models.ExponentialCutoffPowerLaw(
        1 / u.eV, 10.0 * u.TeV, alpha, e_cutoff
    )
    IC = naima.models.InverseCompton(ECPL, seed_photon_fields=["CMB"])
    IC.set_We(We, Eemin=1 * u.TeV)

    # compute flux at the energies given in data['energy']
    model = IC.flux(data, distance=1.0 * u.kpc)

    # Save this realization of the particle distribution function
    elec_energy = np.logspace(11, 15, 100) * u.eV
    nelec = ECPL(elec_energy)

    return model, (elec_energy, nelec)


def IC_We_lnprior(pars):
    logprob = naima.uniform_prior(pars[1], -1, 5)
    return logprob


#
# FUNCTIONAL MODELS
#
# Exponential cutoff powerlaw
# ===========================

ECPL_p0 = np.array((1e-12, 2.4, np.log10(15.0)))
ECPL_labels = ["norm", "index", "log10(cutoff)"]


def ECPL(pars, data):
    # Get the units of the flux data and match them in the model amplitude
    amplitude = pars[0] * data["flux"].unit
    alpha = pars[1]
    e_cutoff = (10 ** pars[2]) * u.TeV
    ECPL = naima.models.ExponentialCutoffPowerLaw(
        amplitude, 1 * u.TeV, alpha, e_cutoff
    )

    return ECPL(data)


def ECPL_lnprior(pars):
    logprob = naima.uniform_prior(pars[0], 0.0, np.inf) + naima.uniform_prior(
        pars[1], -1, 5
    )
    return logprob


# Log-Parabola or Curved Powerlaw
# ===============================

LP_p0 = np.array((1.5e-12, 2.7, 0.12))
LP_labels = ["norm", "alpha", "beta"]


def LP(pars, data):
    amplitude = pars[0] * data["flux"].unit
    alpha = pars[1]
    beta = pars[2]
    LP = naima.models.LogParabola(amplitude, 1 * u.TeV, alpha, beta)
    return LP(data)


def LP_lnprior(pars):
    logprob = naima.uniform_prior(pars[0], 0.0, np.inf) + naima.uniform_prior(
        pars[1], -1, 5
    )
    return logprob