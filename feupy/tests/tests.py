#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""tests classes."""


# In[ ]:


from astropy import units as u
from astropy.units import Quantity

from feupy.target import Target
from feupy.roi import ROI


# In[ ]:


__all__ = [
    "test_target",
    "test_roi"
]


# In[ ]:


def test_target():
    return Target(
        name="23HWC J1825-134", 
        pos_ra=27.46* u.Unit('deg'), 
        pos_dec=12.2* u.Unit('deg'),
    
    )


# In[ ]:


def test_roi():
    return ROI(test_target(), 
        u.Quantity("1.0deg")
    )


# In[ ]:


from feupy.cta.observation import ObservationParameters
from feupy.cta.irfs import CTAOPerf


# In[ ]:


from astropy.coordinates import SkyCoord, Angle
def test_cta_obs_parm():
    return ObservationParameters(
    livetime=50*u.h, 
    offset=0.11*u.deg, 
    e_edges_min=0.1*u.TeV, 
    e_edges_max=100.*u.TeV,
    on_region_radius=Quantity("1.0 deg"),
    n_obs=1000
)


# In[8]:


# from astropy import units as u
# from astropy.coordinates import SkyCoord
# 

# from feupy.scripts import gammapy_catalogs 

# from feupy.catalog.pulsar.atnf import SourceCatalogATNF
# from feupy.catalog.lhaaso import SourceCatalogPublishNatureLHAASO
# from feupy.catalog.hawc import SourceCatalogExtraHAWC

# 


# from gammapy.datasets import FluxPointsDataset
# from gammapy.datasets import Datasets
# from feupy.utils.string_handling import name_to_txt

# from gammapy.modeling.models import SkyModel, Models

# from gammapy.estimators import FluxPoints

