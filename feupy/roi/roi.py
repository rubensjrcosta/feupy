#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""ROI classes."""


# In[1]:


import pandas as pd

from astropy import units as u
from astropy.units import Quantity
 
from feupy.catalog.utils import load_catalogs 

from feupy.catalog.pulsar.atnf import SourceCatalogATNF

from feupy.target import Target

from feupy.utils.coordinates import skcoord_to_dict, dict_to_skcoord

from gammapy.catalog import SourceCatalogObjectHGPS, SourceCatalogObjectGammaCat


# In[ ]:





# In[2]:


__all__ = [
    "ROI",
]


# In[43]:


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
        self.target = target
        self.radius = Quantity(radius, "deg")
        self.dict = self._to_dict()
        self.catalogs = None
        self.pulsars = None
        self.sources = None
        
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

    def _to_dict(self):
        _dict = {}
        _dict["target"] = self.target.dict.copy()
        _dict["radius"] = self.radius
        return _dict

    @property
    def info(self):
        """ROI report (`str`)."""
        ss = 'Target:\n'
        target_info = self.target.info()
        ss += '{}'.format(target_info)
        ss += '\nRegion:\n'
        _ss = "radius={:.2f}\n".format(self.radius).replace(' ', '').replace('=', ' = ')
        ss += _ss
        if self.pulsars is not None:
            ss += f"\nTotal number of pulsars: {len(self.pulsars)}\n"
        if self.sources is not None:
            ss += f"\nTotal number of gamma ray sources: {len(self.sources)}\n"
        return ss
    
    
    def get_catalogs(self, cats="all"):
        
        _catalogs = []
        self.catalogs = []
        self.pulsars = []
        self.sources = []
        
        _catalogs = load_catalogs(cats)
        position = self.target.position 
        radius = self.radius 
        
        for catalog in _catalogs:        
            # Selects only sources within the region of interest. 
            separation = position.separation(catalog.positions)
            mask_roi = separation < radius

            if len(catalog[mask_roi].table):
                self.catalogs.append(catalog[mask_roi])
                for source in catalog[mask_roi]:
                    if catalog[mask_roi].tag == SourceCatalogATNF.tag:
                        self.pulsars.append(source)
                    else: self.sources.append(source)
    
    @staticmethod
    def get_dict_sep(target_pos, sources, opt="pos_skycoord"):
        """
         opt: 'pos_dict' or 'pos_skycoord'  
         """
        dict_sep = {}

        for index, source in enumerate(sources):
            source_name = source.name
            if isinstance(source, SourceCatalogObjectGammaCat):
                source_name = f"{source_name}: gamma-cat"
            if isinstance(source, SourceCatalogObjectHGPS):
                source_name = f"{source_name}: hgps"

            source_pos = source.position
            sep = source_pos.separation(target_pos).deg
            if opt == 'pos_skycoord':
                dict_sep[source_name] = {
                    'position': source_pos,
                    'separation': sep,
                }
            elif opt == 'pos_dict':
                dict_sep[source_name] = {
                    'position': skcoord_to_dict(source_pos),
                    'separation':sep
                }
        return dict_sep
    
    @staticmethod
    def get_df_sep(dict_sep):

        df = pd.DataFrame()
        df["Source name"] = dict_sep.keys()
        col_ra = []
        col_dec = []
        col_sep = []

        for index, name in enumerate(dict_sep.keys()):
            pos = dict_sep[name]["position"]
            if isinstance(pos, dict):
                pos = dict_to_skcoord(pos)
            col_ra.append(pos.ra.deg)
            col_dec.append(pos.dec.deg)
            col_sep.append(dict_sep[name]["separation"])

        df["RA(deg)"] = col_ra
        df["dec.(deg)"] = col_dec
        df["Sep.(deg)"] = col_sep

        return df


    def __repr__(self):
        ss = f"{self.__class__.__name__}("
        ss += f"name={self.target.name!r}, "
        ss += "pos_ra=Quantity('{:.2f}'), ".format(self.target.position.ra).replace(' ', '')
        ss += "pos_dec=Quantity('{:.2f}'), ".format(self.target.position.dec).replace(' ', '')
        ss += "radius=Quantity('{:.2f}'))\n".format(self.radius).replace(' ', '')
        return ss.replace('=', ' = ')   


# In[ ]:




