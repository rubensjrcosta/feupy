#!/usr/bin/env python
# coding: utf-8

# [gammapy.modeling.models.SpectralModel](https://docs.gammapy.org/1.0/api/gammapy.modeling.models.SpectralModel.html#gammapy.modeling.models.SpectralModel)

# In[1]:


from feupy.config import *


# In[2]:


from feupy.utils.string_handling import name_to_txt


# In[3]:


from astropy import units as u
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel

def sky_model_pl(
    index = 2,
    amplitude = 1e-12 * u.Unit("TeV-1 cm-2 s-1"),
    reference = 10 * u.Unit("TeV"),
    datasets_names = None
):
    """
    Returns a sky model with spectral model type: PowerLawSpectralModel
    >>>sky_model = sky_model_pl(
    index = 2,
    amplitude = 1e-12 * u.Unit("TeV-1 cm-2 s-1"),
    reference = 1 * u.Unit("TeV"),
    name = "pl",
)
    >>>print(sky_model)
    SkyModel

      Name                      : lp
      Datasets names            : None
      Spectral model type       : LogParabolaSpectralModel
      Spatial  model type       : 
      Temporal model type       : 
      Parameters:
        amplitude                     :   1.40e-12   +/- 6.7e-14 1 / (cm2 s TeV)
        reference             (frozen):      1.000       TeV         
        alpha                         :      1.577   +/-    0.03             
        beta                          :      0.233   +/-    0.01  
    """
    
    spectral_model = PowerLawSpectralModel(
        index=index, 
        amplitude=amplitude, 
        reference=reference,
    )
            
    if not datasets_names:
        sky_model = SkyModel(
            spectral_model = spectral_model,
        )
    else:
        sky_model = SkyModel(
            spectral_model = spectral_model, 
            name = f"{name_to_txt(datasets_names)}_{spectral_model.tag[1]}",
            datasets_names = datasets_names         
        )
    return sky_model


# In[1]:


from astropy import units as u
from gammapy.modeling.models import ExpCutoffPowerLawSpectralModel, SkyModel

def sky_model_ecpl(
    amplitude = 1e-12 * u.Unit("TeV-1 cm-2 s-1"),
    index = 2,
    lambda_= 0.1 * u.Unit("TeV-1"),
    reference = 10 * u.Unit("TeV"),
    alpha = 1.0,
    datasets_names = None
):
    """
    Returns a sky model with spectral model type: ExpCutoffPowerLawSpectralModel
    see: https://docs.gammapy.org/1.1/user-guide/model-gallery/spectral/plot_exp_cutoff_powerlaw.html
    
    >>>sky_model = sky_model_ecpl(
    amplitude = 1e-12 * u.Unit("TeV-1 cm-2 s-1"),
    index = 2,
    lambda_= 0.1 * u.Unit("TeV-1"),
    reference = 1 * u.Unit("TeV"),
    alpha = 1.0,
    name = "ecpl",
)
    >>>print(sky_model)
    SkyModel

      Name                      : lp
      Datasets names            : None
      Spectral model type       : LogParabolaSpectralModel
      Spatial  model type       : 
      Temporal model type       : 
      Parameters:
        amplitude                     :   1.40e-12   +/- 6.7e-14 1 / (cm2 s TeV)
        reference             (frozen):      1.000       TeV         
        alpha                         :      1.577   +/-    0.03             
        beta                          :      0.233   +/-    0.01  
    """
    
    spectral_model = ExpCutoffPowerLawSpectralModel(
        amplitude = amplitude,
        index = index,
        lambda_= lambda_,
        reference = reference,
        alpha = alpha,
    )
    if not datasets_names:
        sky_model = SkyModel(
            spectral_model = spectral_model,
        )    
    else:
        name = f"{name_to_txt(datasets_names)}_{spectral_model.tag[1]}"
        sky_model = SkyModel(
            spectral_model = spectral_model, 
            name = name,
            datasets_names = datasets_names
        )
    return sky_model


# In[ ]:





# In[ ]:


from astropy import units as u
from gammapy.modeling.models import LogParabolaSpectralModel, SkyModel

def sky_model_lp(
    alpha = 2.3,
    amplitude = 1e-12 * u.Unit("TeV-1 cm-2 s-1"),
    reference = 1 * u.Unit("TeV"),
    beta = 0.5,
    datasets_names = None
):
    """
    Returns a sky model with spectral model type: LogParabolaSpectralModel
    """
    
    spectral_model = LogParabolaSpectralModel(
        alpha = alpha,
        amplitude = amplitude,
        reference = reference,
        beta = beta,
    )
    if not datasets_names:
        sky_model = SkyModel(
            spectral_model = spectral_model,
        )    
    else:
        name = f"{name_to_txt(datasets_names)}_{spectral_model.tag[1]}"
        sky_model = SkyModel(
            spectral_model=spectral_model, 
            name= name,
            datasets_names = datasets_names
        )
    return sky_model


# In[ ]:


from astropy import units as u
import matplotlib.pyplot as plt
from gammapy.modeling.models import BrokenPowerLawSpectralModel, Models, SkyModel

def sky_model_bpl(
    index1=1.5,
    index2=2.5,
    amplitude=1e-12 * u.Unit("TeV-1 cm-2 s-1"),
    ebreak=1 * u.Unit("TeV"),
    datasets_names = None
):
    """
    Returns a sky model with spectral model type: BrokenPowerLawSpectralModel
    """
    
    spectral_model =BrokenPowerLawSpectralModel(
        index1=index1,
        index2=index2,
        amplitude=amplitude,
        ebreak=ebreak,
    )
    if not datasets_names:
        sky_model = SkyModel(
            spectral_model = spectral_model,
        )    
    else:
        name = f"{name_to_txt(datasets_names)}_{spectral_model.tag[1]}"
        sky_model = SkyModel(
            spectral_model=spectral_model, 
            name= name,
            datasets_names = datasets_names 
        )
    return sky_model

