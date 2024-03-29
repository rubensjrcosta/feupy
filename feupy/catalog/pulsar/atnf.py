#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""ATNF Pulsar Catalogue and source classes."""


# In[ ]:


from astropy.table import Table


from gammapy.utils.scripts import make_path
from gammapy.catalog.core import SourceCatalog, SourceCatalogObject


# In[ ]:


__all__ = [
    "SourceCatalogATNF",
    "SourceCatalogObjectATNF",
]


# In[1]:


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
    
#     def __str__(self):
#         return self.info()
    
    @property
    def dist(self):
        """
        """
        return self.data['DIST']
    
    @property
    def age(self):
        """
        """
        return self.data['AGE']

    @property
    def P_0(self):
        """
        """
        return self.data['P0']
    
    @property
    def B_surf(self):
        """
        """
        return self.data['BSURF']
    
    @property
    def E_dot(self):
        """
        """
        return self.data['EDOT']
    
    @property
    def Type(self):
        """
        """
        return self.data['TYPE']
    
    @property
    def assoc(self):
        """
        """
        return self.data['ASSOC']
    
class SourceCatalogATNF(SourceCatalog):
    """ATNF Pulsar Catalogue.

    See: https://www.atnf.csiro.au/research/pulsar/psrcat/

    One source is represented by `~feupy.pulsar.atnf.SourceCatalogATNF`.
    """  
      
    tag = "ATNF"
    """Catalog name"""
        
    description = "A comprehensive database of all published pulsars"
    """Catalog description"""
    
    PSR_PARAMS = [
        'JNAME', 
        'RAJD', 
        'DECJD',
        'RAJ', 
        'DECJ',
        'DIST',
        'DIST_DM', 
        'AGE', 
        'P0',
        'BSURF',
        'EDOT', 
        'TYPE', 
        'ASSOC',
    ]
    """Pulsar default parameters
    
    See: https://www.atnf.csiro.au/research/pulsar/psrcat/psrcat_help.html?type=normal&highlight=name#par_list
    """
    
    source_object_class = SourceCatalogObjectATNF
    
    def __init__(self, filename="$PYTHONPATH/data/catalogs/ATNF/atnf.fits"):
        table = Table.read(make_path(filename))
        source_name_key = "JNAME"
        super().__init__(table=table, source_name_key=source_name_key)

