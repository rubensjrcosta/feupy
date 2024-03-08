#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""ATNF Pulsar Catalogue and source classes."""


# In[2]:


from astropy.table import Table


from gammapy.utils.scripts import make_path
from gammapy.catalog.core import SourceCatalog, SourceCatalogObject


# In[3]:


__all__ = [
    "SourceCatalogATNF",
    "SourceCatalogObjectATNF",
]


# In[ ]:





# In[28]:


class SourceCatalogObjectATNF(SourceCatalogObject):
    """One source from the ATNF Pulsar Catalogue.

    See: Manchester, R. N., Hobbs, G. B., Teoh, A. & Hobbs, M., Astron. J., 129, 1993-2006 (2005) (astro-ph/0412641)
    
    
    The data are available through the web page (http://www.atnf.csiro.au/research/pulsar/psrcat) 
    in the section ‘Public Data’. 

    One source is represented by `~feupy.catalog.pulsar.SourceCatalogObjectATNF`.
    """    
    _source_name_key = "JNAME"
    
#     def __repr__(self):
#         ss = f"{self.__class__.__name__}("
#         ss += f"name={self.name!r}, "
#         ss += "pos_ra=Quantity('{:.2f}'), ".format(self.position.ra).replace(' ', '')
#         ss += "pos_dec=Quantity('{:.2f}'), ".format(self.position.dec).replace(' ', '')
#         ss += "dist=Quantity('{:.2f}'), ".format(self.dist).replace(' ', '')
#         ss += "age=Quantity('{:.2e}'), ".format(self.age).replace(' ', '')
#         ss += "P_0=Quantity('{:.2f}'), ".format(self.P_0).replace(' ', '')
#         ss += "B_surf=Quantity('{:.2e}'), ".format(self.B_surf).replace(' ', '')
#         ss += "E_dot=Quantity('{:.2e}'), ".format(self.E_dot).replace(' ', '')
#         ss += "Type={:<20s}, ".format(self.Type)
#         ss += "assoc={:<20s})\n".format(self.assoc)
#         return ss  
    
    def __str__(self):
        return self.info()
    
    def info(self, info="all"):
        """Summary information string.

        Parameters
        ----------
        info : {'all', 'basic', 'position', 'timing-profile, 'distance', 'associations-survey', 'derived'}
            Comma separated list of options
        """
        if info == "all":
#             info = "basic,position,timing-profile,binary"
            info = "basic,position,timing-profile,distance,associations-survey,derived"

        ss = ""
        ops = info.split(",")
        if "basic" in ops:
            ss += self._info_basic()
        if "position" in ops:
            ss += self._info_position()
        if "timing-profile" in ops:
            ss += self._info_timing_profile()
#         if "binary" in ops:
#             ss += self._info_binary()
        if "distance" in ops:
            ss += self._info_distance()
        if "associations-survey" in ops:
            ss += self._info_associations_survey()
        if "derived" in ops:
            ss += self._info_derived()
            
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
            f"RA: {self.data.RAJ2000:.3f} +- {self.data.RAJ2000_ERR:.3f}\n"
            f"DEC: {self.data.DEJ2000:.3f} +- {self.data.DEJ2000_ERR:.3f}\n"
        )
    
    def _info_timing_profile(self):
        """ Print timing solution and profile parameters info."""
        ss = "\n*** Timing and profile info ***\n\n"
        ss += f"P0: {self.data.P0.value:.3e} +- {self.data.P0_ERR:.3e}\n"
        
        return ss

#     def _info_binary(self):
#         """ Binary system parameters info."""
#         ss = "\n*** Binary system info ***\n\n"
#         ss += f"Binary: {self.data.Binary:.2e} (Binary model)\n"
#         return ss
    
    def _info_distance(self):
        """ Print distance parameters info."""
        ss = "\n*** Distance info ***\n\n"
        ss += f"Dist: {self.data.DIST:.2e}\n"
        ss += f"Dist_DM: {self.data.DIST_DM:.2e}\n"
        return ss
    
    def _info_associations_survey(self):
        """ Print associations and survey parameters info."""
        ss = "\n*** Associations and survey info ***\n\n"
        ss += f"Assoc: {self.data.ASSOC:.2s}\n"
        ss += f"Type: {self.data.TYPE:.2s}\n"
        return ss
    
    def _info_derived(self):
        """ Print derived parameters info."""
        ss = "\n*** Derived parameters info ***\n\n"
        ss += f"Age: {self.data.AGE:.2e}\n"
        ss += f"BSurf: {self.data.BSURF:.2e}\n"
        ss += f"Edot: {self.data.EDOT:.2e}\n"
        return ss
    
class SourceCatalogATNF(SourceCatalog):
    """ATNF Pulsar Catalogue.

    See: https://www.atnf.csiro.au/research/pulsar/psrcat/

    One source is represented by `~feupy.pulsar.atnf.SourceCatalogATNF`.
    """  
          
    tag = "ATNF"
    """Catalog name"""
        
    description = "A comprehensive database of all published pulsars"
    """Catalog description"""
        
    source_object_class = SourceCatalogObjectATNF
    
    def __init__(self, filename="$PYTHONPATH/data/catalogs/ATNF/atnf.fits"):
        table = Table.read(make_path(filename))
            
        """Pulsar default parameters
        See: https://www.atnf.csiro.au/research/pulsar/psrcat/psrcat_help.html?type=normal&highlight=name#par_list
        """
        source_name_key = "JNAME"
        super().__init__(table=table, source_name_key=source_name_key)
    
    @property
    def PSR_PARAMS(self):
        return self.table.colnames
    
    @property
    def PSR_PARAMS_DESCRIPTION(self):
        ss = "\n*** The Pulsar Parameters ***\n\n" 
        for par in self.PSR_PARAMS:
            if self.table[par].unit is not None:
                unit = f" ({self.table[par].unit})"
            else: unit = ""
            ss += f"{self.table[par].name}: {self.table[par].description}{unit}\n"
        return print(ss)
        


# In[ ]:




