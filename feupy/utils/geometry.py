#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gammapy.data import Observation
from gammapy.maps import RegionGeom

from regions import CircleSkyRegion

from astropy.coordinates import SkyCoord

from astropy import units as u
from gammapy.maps import MapAxis


# In[2]:


__all__ = [
    "create_energy_axis",
    "set_pointing",
#     "create_observation",
    "define_on_region",
    "create_region_geometry",
]


# In[3]:


def create_energy_axis(energy_min, energy_max, nbin=5, per_decade=True, name="energy"):    
    return MapAxis.from_energy_bounds(
        energy_min=energy_min, 
        energy_max=energy_max, 
        nbin=nbin, 
        per_decade=per_decade, 
        name=name
    )


# In[4]:


def set_pointing(position, offset):
    """Set the pointing position of the observation"""
    return SkyCoord(position.ra, position.dec + offset)


# In[5]:


def define_on_region(center, radius):
    """
    on_region_radius :Angle()
    """
    return CircleSkyRegion(
        center=center, 
        radius=radius
    )


# In[ ]:


def create_region_geometry(on_region, axes):
    """Defines the geometry"""
    return RegionGeom.create(
        region=on_region, 
        axes=axes
    )


# In[ ]:




