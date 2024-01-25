#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst


# In[2]:


import numpy as np
import pandas as pd 

from feupy.roi import ROI
from feupy.target import Target
from feupy.utils.string_handling import name_to_txt
from feupy.scripts import gammapy_catalogs 

from feupy.catalog.pulsar.atnf import SourceCatalogATNF
from feupy.catalog.lhaaso import SourceCatalogPublishNatureLHAASO
from feupy.catalog.hawc import SourceCatalogExtraHAWC

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.units import Quantity

from gammapy.utils.units import energy_unit_format

from gammapy.datasets import FluxPointsDataset
from gammapy.datasets import Datasets

from gammapy.modeling.models import SkyModel, Models

from gammapy.estimators import FluxPoints

from pathlib import Path


# In[4]:


__all__ = ["CounterpartsAnalysisConfig"]


# In[5]:


# How to create a class:
class CounterpartsAnalysisConfig:
   # ADD others parameters
   
    # color="red" # The color of the flux ponts
    all=[]
    @u.quantity_input(pos_ra=u.deg, pos_dec=u.deg, radius=u.deg, e_ref_min=u.eV, e_ref_max=u.eV)
    def __init__(self, target_name: str, pos_ra, pos_dec, radius, e_ref_min=None, e_ref_max=None, catalogs_roi=None):
       # Run validations to the received arguments
        assert  0 <= pos_ra.value <= 360, f"Right Ascension {pos_ra} is not in the range: (0,360) deg!"
        assert -90 <= pos_dec.value <= 90, f"Declination {pos_dec} is not in the range: (-90,90) deg!"

        # Assign to self object
        self.__target_name = target_name
        self.position = SkyCoord(pos_ra, pos_dec) 
        self.radius = radius
        if e_ref_min is not None:
            self.e_ref_min = Quantity(e_ref_min, "TeV")
        else: self.e_ref_min = e_ref_min
        if e_ref_max is not None:
            self.e_ref_max = Quantity(e_ref_max, "TeV")
        else: self.e_ref_max = e_ref_max
        self.energy_range = [self.e_ref_min, self.e_ref_max]
        self.target = Target(self.__target_name, self.position.ra, self.position.dec)
        self.roi = ROI(self.__target_name, self.position.ra, self.position.dec, self.radius)
        
        # Actions to execute
        CounterpartsAnalysisConfig.all.append(self)
    
    @property
    # Property Decorator=Read-Only Attribute
    def info(self):
        info={}
        info["target_name"] = self.target_name
        info["position"] = self.position
        info["radius"] = self.radius
        info["energy_range"] = self.energy_range
        return info    
    
    @property
    def target_name(self):
        return self.__target_name

    def __repr__(self):
        ss = f"{self.__class__.__name__}("
        ss += f"target_name={self.__target_name}, "
        ss += "pos_ra=Quantity('{:.2f}'), ".format(self.position.ra).replace(' ', '')
        ss += "pos_dec=Quantity('{:.2f}'), ".format(self.position.dec).replace(' ', '')
        ss += "radius=Quantity('{:.2f}'), ".format(self.radius).replace(' ', '')
        if self.e_ref_min is None: ss += "e_ref_min=None, "
        else: ss += "e_ref_min=Quantity('{}'), ".format(energy_unit_format(self.e_ref_min).replace(' ', ''))
        if self.e_ref_max is None: ss += "e_ref_max=None)"
        else: ss += "e_ref_max=Quantity('{}'))".format(energy_unit_format(self.e_ref_max).replace(' ', ''))
        return ss 


# In[7]:


def test_analysis_confg():
    return CounterpartsAnalysisConfig(
        "LHAASO J1825-1326", 
        276.45* u.Unit('deg'), 
        -13.45* u.Unit('deg'),
        1* u.Unit('deg'),
        1* u.Unit('erg')
    )


# In[10]:





# In[ ]:




