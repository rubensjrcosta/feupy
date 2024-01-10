#!/usr/bin/env python
# coding: utf-8

# In[2]:





# In[1]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst

from psrqpy import QueryATNF
def query_ATNF():
    _table = QueryATNF
    return _table 

"""Pulsar default parameters"""
PSR_PARAMS =['JNAME', 'RAJD', 'DECJD','RAJ', 'DECJ','DIST','DIST_DM', 'AGE', 'P0','BSURF','EDOT', 'TYPE', 'Assoc']


class SourceCatalogATNF():
    """ATNF Pulsar Catalogue.

    See: https://www.atnf.csiro.au/research/pulsar/psrcat/

    One source is represented by `~feupy.catalog.SourceCatalogATNF`.
    """    
    tag = "atnf"
    description = "An online catalog of pulsars"
    
    
    def PSR_PARAMS(self):
        return PSR_PARAMS
    
    
    def __init__(self):
        self.__query = query_ATNF()

    def table(self):
        return self.__query().table

    def pandas(self):
        return self.__query().pandas

    @property
    # Property Decorator = Read-Only Attribute
    def query(self):
        return self.__query
    

    @property        
    def version(self):
        return __query.get_version


# In[ ]:




