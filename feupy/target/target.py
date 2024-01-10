#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from astropy import units as u
from astropy.units import Quantity
from astropy.coordinates import SkyCoord
from gammapy.utils.units import energy_unit_format


# In[2]:


__all__ = [
#     "ROI",
    "Target",
#     "AnalysisConfig",
]


# In[5]:


class Target:
    """Target information."""
    
    all = []
    # Validating the units of arguments to functions
    @u.quantity_input(pos_ra= u.deg,  pos_dec= u.deg)
    def __init__(self, name: str, pos_ra,  pos_dec):

        # Run validations to the received arguments
        assert  0 <= pos_ra.value <= 360, f"Right Ascension {pos_ra} is not in the range: (0,360) deg!"
        assert -90 <= pos_dec.value <= 90, f"Declination {pos_dec} is not in the range: (-90,90) deg!"
        
        # Assign to self object
        self.__name = name
        self.position = SkyCoord(pos_ra,pos_dec) # convert coordinates to astropy SkyCoord
        
        # Actions to execute
        Target.all.append(self) 
        
    @property
    def info(self):
        info = {}
        info["name"] = self.name
        info["position"] = self.position
        return info
    
    @property
    # Property Decorator = Read-Only Attribute
    def name(self):
        return self.__name
    
    def __repr__(self):
        return  f"{self.__class__.__name__}({self.__name}, {(self.position.ra.value)}.deg, {(self.position.dec.value)}.deg)"


# In[8]:


def test_target():
    return Target(
        "2HWC J1825-134", 
        27.46* u.Unit('deg'), 
        12.2* u.Unit('deg')
    )

    

