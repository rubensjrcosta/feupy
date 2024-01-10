#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# from .maker import get_catalogs
from astropy import units as u
from astropy.units import Quantity
from astropy.coordinates import SkyCoord
from gammapy.utils.units import energy_unit_format


# In[2]:


__all__ = [
    "ROI",
#     "Target",
#     "AnalysisConfig",
]


# In[4]:


class ROI:
    # ADD others parameters
    all=[]

    # Validating the units of arguments to functions
    @u.quantity_input(pos_ra=u.deg, pos_dec=u.deg, radius=u.deg)
    def __init__(self, name: str, pos_ra, pos_dec, radius):

        # Run validations to the received arguments
        assert 0 <= pos_ra.value <= 360, f"Right Ascension {pos_ra} is not in the range: (0,360) deg!"
        assert -90 <= pos_dec.value <= 90, f"Declination {pos_dec} is not in the range: (-90,90) deg!"

        # Assign to self object
        self.__name=name
        self.radius=radius
        self.position=SkyCoord(pos_ra,pos_dec) # convert coordinates to astropy SkyCoord

        # Actions to execute
        ROI.all.append(self) 

    @property
    def info(self):
        info={}
        info["name"]=self.__name
        info["position"]=self.position
        info["radius"]=self.radius
        return info

    @property
    # Property Decorator=Read-Only Attribute
    def name(self):
        return self.__name

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__name}, {(self.position.ra.value)}.deg, {(self.position.dec.value)}.deg, {(self.radius.value)}.deg)"


# In[7]:


def test_roi():
    return ROI(
        "LHAASO J1825-1326", 
        u.Quantity("276.45deg"), 
        -13.45* u.Unit('deg'), 
        u.Quantity("1.0deg")
    )

