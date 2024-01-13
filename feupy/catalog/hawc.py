#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""ExtraHAWC catalog and source classes."""


# In[2]:


import pickle

# from feupy.source import Source
# from feupy.utils.string_handling import *
from feupy.utils.table import remove_nan

from astropy.table import Table

from gammapy.utils.scripts import make_path
from gammapy.estimators import FluxPoints
from gammapy.modeling.models import SkyModel
from gammapy.catalog.core import SourceCatalog, SourceCatalogObject


# In[ ]:


__all__ = [
    "SourceCatalogExtraHAWC",
    "SourceCatalogObjectExtraHAWC",
]


# In[5]:


class SourceCatalogObjectExtraHAWC(SourceCatalogObject):
    """One source from the HAWC Catalogue.

    See: ...
    
    The data are available through the web page (http://english.ihep.cas.cn/lhaaso/index.html) 
    in the section ‘Public Data’. 

    One source is represented by `~feupy.catalog.SourceCatalogExtraHAWC`.
    """    
    _source_name_key = "source_name"
    
    _filename="$PYTHONPATH/feupy/data/catalogs/extraHAWC/extraHAWC.pkl"    
    with open(make_path(_filename), "rb") as fp:  
        _data = pickle.load(fp)
    
    def __repr__(self):
        ss = f"{self.__class__.__name__}("
        ss += f"name={self.name!r}, "
        ss += "pos_ra=Quantity('{:.2f}'), ".format(self.position.ra).replace(' ', '')
        ss += "pos_dec=Quantity('{:.2f}'))\n".format(self.position.dec).replace(' ', '')
        return ss  

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

    
class SourceCatalogExtraHAWC(SourceCatalog):
    """HAWC  Catalogue.

    See: https://doi.org/10.1038/s41586-021-03498-z

    One source is represented by `~feupy.catalog.SourceCatalogExtraHAWC`.
    """    
    tag = "extraHAWC"
    """Catalog name"""
        
    description = "extraHAWC catalog from the HAWC observatory"
    """Catalog description"""
    
    source_object_class = SourceCatalogObjectExtraHAWC
    
    def __init__(self, filename="$PYTHONPATH/feupy/data/catalogs/extraHAWC/extraHAWC.fits"):
        table = Table.read(make_path(filename))
        source_name_key = "Source_Name"
        super().__init__(table=table, source_name_key=source_name_key)


# In[ ]:




