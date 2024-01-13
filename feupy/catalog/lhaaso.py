#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pickle

# from feupy.source import Source
# from feupy.utils.string_handling import *
from feupy.utils.table import remove_nan

from astropy.table import Table

from gammapy.modeling.models import SkyModel

from gammapy.utils.scripts import make_path
from gammapy.estimators import FluxPoints
from gammapy.catalog.core import SourceCatalog, SourceCatalogObject


# In[ ]:


__all__ = [
    "SourceCatalogPublishNatureLHAASO",
    "SourceCatalogObjectPublishNatureLHAASO",
]


# In[ ]:





# In[11]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Source catalog for LHAASO"""
class SourceCatalogObjectPublishNatureLHAASO(SourceCatalogObject):
    """One source from the LHAASO first 12 PeVatrons Catalogue.

    See: https://doi.org/10.1038/s41586-021-03498-z
    
    The data are available through the web page (http://english.ihep.cas.cn/lhaaso/index.html) 
    in the section ‘Public Data’. 

    One source is represented by `~feupy.catalog.SourceCatalogLHAASO`.
    """    
    _source_name_key = "source_name"
    
    _filename="$PYTHONPATH/feupy/data/catalogs/publishNatureLHAASO/publishNatureLHAASO.pkl"    
    with open(make_path(_filename), "rb") as fp:  
        _data = pickle.load(fp)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r})"

    def __str__(self):
        return self.info()
    
    def spectral_model(self):
        """Spectral model as a `~gammapy.modeling.models.SpectralModel` object."""
        return self._data[self.name]['spectral_model']
    
    def sky_model(self):
        """Source sky model (`~gammapy.modeling.models.SkyModel`)."""
        return SkyModel(
        #             spatial_model=self.spatial_model(),
            spectral_model=self.spectral_model(),
            name=self.name,
        )
            
    @property
    def flux_points(self):
        """Flux points (`~gammapy.estimators.FluxPoints`)."""
        return FluxPoints.from_table(
            table=self.flux_points_table,
            reference_model=self.sky_model(),
            sed_type='e2dnde',
        )
    
    @property
    def flux_points_table(self):
        """Flux points table as a `~astropy.table.Table`."""
        table = Table()
        table.meta["sed_type_init"] = "e2dnde"
        table["e_ref"] = self.data["e_ref"]
        table["e2dnde"] = self.data["e2dnde"]
        table["e2dnde_errn"] = self.data["e2dnde_errn"]
        table["e2dnde_errp"] = self.data["e2dnde_errp"]
        table["is_ul"] = self.data["is_ul"]
        return remove_nan(table)
    
    
class SourceCatalogPublishNatureLHAASO(SourceCatalog):
    """LHAASO first 12 PeVatrons Catalogue.

    See: https://doi.org/10.1038/s41586-021-03498-z
    
    The data are available through the web page (http://english.ihep.cas.cn/lhaaso/index.html) 
    in the section ‘Public Data’. 

    One source is represented by `~feupy.catalog.SourceCatalogLHAASO`.
    """    
    tag = "lhaaso-nature"
    """Catalog name"""
        
    description = "LHAASO first 12 PeVatrons Catalogue"
    """Catalog description"""
    
    source_object_class = SourceCatalogObjectPublishNatureLHAASO
    
    def __init__(self, filename="$PYTHONPATH/feupy/data/catalogs/publishNatureLHAASO/publishNatureLHAASO.fits"):
        table = Table.read(make_path(filename))
        source_name_key = "Source_Name"
        super().__init__(table=table, source_name_key=source_name_key)


# In[ ]:





# In[ ]:




