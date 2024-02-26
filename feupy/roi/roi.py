#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""ROI classes."""


# In[2]:


from astropy import units as u
from astropy.units import Quantity

from feupy.scripts import gammapy_catalogs 

from feupy.catalog.pulsar.atnf import SourceCatalogATNF
from feupy.catalog.lhaaso import SourceCatalogPublishNatureLHAASO
from feupy.catalog.hawc import SourceCatalogExtraHAWC

from feupy.target import Target


# In[3]:


__all__ = [
    "ROI",
]


# In[4]:


class ROI:
    # ADD others parameters
    all=[]

    # Validating the units of arguments to functions
    @u.quantity_input(radius=u.deg)
    def __init__(self, 
                 target, 
                 radius
                ):

        # Assign to self object
        self.target=target
        self.radius=Quantity(radius, "deg")
        
        # Actions to execute
        ROI.all.append(self) 
        
    @property
    def target(self):
        """Target as an `~feupy.target.Target` object."""
        return self._target

    @target.setter
    def target(self, value):
        if isinstance(value, Target):
            self._target = value
        else:
            raise TypeError("target must be Target")

    @property
    def info(self):
        """ROI report (`str`)."""
        ss = 'Target:\n'
        target_info = self.target.info
        ss += '{}'.format(target_info)
        ss += 'Region:\n'
        _ss = "radius={:.2f}\n".format(self.radius).replace(' ', '').replace('=', ' = ')
        ss += _ss
        return ss
    
    @property
    def catalogs(self):
        _catalogs = []
        catalogs_roi = []
        sources = [] 
        pulsars = [] 
        
        position = self.target.position 
        radius = self.radius 

        _catalogs.extend(gammapy_catalogs.load_all_catalogs())
        _catalogs.append(SourceCatalogExtraHAWC())
        _catalogs.append(SourceCatalogPublishNatureLHAASO())
        _catalogs.append(SourceCatalogATNF())

        for catalog in _catalogs:        
            # Selects only sources within the region of interest. 
            separation = position.separation(catalog.positions)

            mask_roi = separation < radius

            if len(catalog[mask_roi].table):
                catalogs_roi.append(catalog[mask_roi])
                for source in catalog[mask_roi]:
                    if catalog[mask_roi].tag == "ATNF":
                        pulsars.append(source)
                    else: sources.append(source)
                       
        self.pulsars = pulsars
        self.sources = sources
#         if info:
#             print(f"Total number of gamma ray sources: {len(sources)}")
#             print(f"Total number of pulsars: {len(pulsars)}")
 
        return catalogs_roi
    
    
    def __repr__(self):
        ss = f"{self.__class__.__name__}("
        ss += f"name={self.target.name!r}, "
        ss += "pos_ra=Quantity('{:.2f}'), ".format(self.target.position.ra).replace(' ', '')
        ss += "pos_dec=Quantity('{:.2f}'), ".format(self.target.position.dec).replace(' ', '')
        ss += "radius=Quantity('{:.2f}'))\n".format(self.radius).replace(' ', '')
        return ss.replace('=', ' = ')   


# In[5]:


# # from feupy.target import Target

# from astropy import units as u
# from astropy.units import Quantity
# from gammapy.modeling.models import (
#     PowerLawSpectralModel,
#     SkyModel,
# )
# from astropy.coordinates import Angle

# name = "LHAASO J1825-1326"
# pos_ra = u.Quantity("276.45deg") 
# pos_dec = -13.45* u.Unit('deg')

# on_region_radius = on_region_radius=Angle("1.0 deg")
# model = PowerLawSpectralModel()
# target = Target(name, pos_ra, pos_dec, spectral_model=model)
# roi = ROI(target, radius=on_region_radius)


# In[ ]:





# In[ ]:





# In[ ]:




