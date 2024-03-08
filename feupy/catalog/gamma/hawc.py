#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""ExtraHAWC catalog and source classes."""


# In[1]:


import pickle

# from feupy.source import Source
# from feupy.utils.string_handling import *
from feupy.utils.table import remove_nan

from astropy.table import Table

from gammapy.utils.scripts import make_path
from gammapy.estimators import FluxPoints
from gammapy.modeling.models import SkyModel
from gammapy.catalog.core import SourceCatalog, SourceCatalogObject


# In[2]:


__all__ = [
    "SourceCatalogExtraHAWC",
    "SourceCatalogObjectExtraHAWC",
]


# In[3]:


class SourceCatalogObjectExtraHAWC(SourceCatalogObject):
    """One source from the HAWC Catalogue.

    See: ...
    
    The data are available through the web page (http://english.ihep.cas.cn/lhaaso/index.html) 
    in the section ‘Public Data’. 

    One source is represented by `~feupy.catalog.SourceCatalogExtraHAWC`.
    """    
    _source_name_key = "source_name"
    
    _filename="$PYTHONPATH/data/catalogs/extraHAWC/extraHAWC.pkl"    
    with open(make_path(_filename), "rb") as fp:  
        _data = pickle.load(fp)
        
    def __str__(self):
        return self.info()
    
    def info(self, info="all"):
        """Summary information string.

        Parameters
        ----------
        info : {'all', 'basic', 'position', 'spectrum'}
            Comma separated list of options
        """
        if info == "all":
            info = "basic,position,spectrum"

        ss = ""
        ops = info.split(",")
        if "basic" in ops:
            ss += self._info_basic()
        if "position" in ops:
            ss += self._info_position()
        if "spectrum" in ops:
            ss += self._info_spectrum()

        return ss

    def _info_basic(self):
        """Print basic information."""
        return (
            f"\n*** Basic info ***\n\n"
            f"Catalog row index (zero-based) : {self.row_index}\n"
            f"Source name : {self.name}\n"
        )
    
    def _info_position(self):
        """Print position information."""
        return (
            f"\n*** Position info ***\n\n"
            f"RA: {self.data.ra:.3f}\n"
            f"DEC: {self.data.dec:.3f}\n"
#             f"GLON: {self.data.glon:.3f}\n"
#             f"GLAT: {self.data.glat:.3f}\n"
#             f"Position error: {self.data.pos_err:.3f}\n"
        )
    
    def _info_spectrum(self):
        """Print spectral info."""
        ss = "\n*** Spectral info ***\n\n"
        if self.spectral_model is not None:
            model = self.spectral_model()
            parameters = model.parameters
            ss += f"Spectrum type:  {model.tag[0]}\n"
            for par in parameters:
                name = par.name
                val = par.value
                err = par.error
                
                try:
                    unit = f"{par.unit:unicode}"
                except: unit = ""

                ss += f"{name}: {val:.3} +- {err} {unit}\n"
    
        else:
            ss += "No spectrum available"

        return ss
    
#     def __repr__(self):
#         ss = f"{self.__class__.__name__}("
#         ss += f"name={self.name!r}, "
#         ss += "pos_ra=Quantity('{:.2f}'), ".format(self.position.ra).replace(' ', '')
#         ss += "pos_dec=Quantity('{:.2f}'))\n".format(self.position.dec).replace(' ', '')
#         return ss  

#     def __str__(self):
#         return self.info()
    
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
    
    def __init__(self, filename="$PYTHONPATH/data/catalogs/extraHAWC/extraHAWC.fits"):
        table = Table.read(make_path(filename))
        source_name_key = "source_name"
        super().__init__(table=table, source_name_key=source_name_key)


# In[ ]:




