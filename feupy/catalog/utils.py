#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Catalog utilities classes."""


# In[6]:


from feupy.catalog import CATALOG_REGISTRY 
from feupy.catalog import CATALOG_REGISTRY_GAMMAPY 
from feupy.catalog import CATALOG_REGISTRY_FEUPY 


# In[7]:


def load_catalog(tag=None):
    """
    """
    return  CATALOG_REGISTRY.get_cls(tag)()


def _get_catalogs(catalogs_registry = None):
    source_catalogs = []
    for index, catalog in enumerate(catalogs_registry):
        catalog_cls = catalogs_registry.get_cls(catalog.tag)()
        source_catalogs.append(catalog_cls)
    return source_catalogs
    

def load_catalogs(cats="all"):
    """load catalogs.

    Parameters
    ----------
    cats : {'all', 'gamma', 'gammapy', 'feupy'}
        list of possible options
    """
    if cats == "all":
        return _get_catalogs(catalogs_registry = CATALOG_REGISTRY)
    elif cats == "gammapy":
        return _get_catalogs(catalogs_registry = CATALOG_REGISTRY_GAMMAPY)
    elif cats == "feupy":
        return _get_catalogs(catalogs_registry = CATALOG_REGISTRY_FEUPY)
    elif cats == "gamma":
        source_catalogs =  _get_catalogs(catalogs_registry = CATALOG_REGISTRY)
        source_catalogs.remove(source_catalogs[-1])
        return source_catalogs

# In[14]:


def catalogs_info():
    """
    """
    print (f"Source catalogs in Gammapy: {len(CATALOG_REGISTRY_GAMMAPY)}\n")
    for gindex, catalog in enumerate(CATALOG_REGISTRY_GAMMAPY):
        print(f"(catalog index: {gindex}) {CATALOG_REGISTRY_GAMMAPY.get_cls(catalog.tag)()}")
    print (f"Source catalogs in Feupy: {len(CATALOG_REGISTRY_FEUPY)}\n")
    for index, catalog in enumerate(CATALOG_REGISTRY_FEUPY):
        print(f"(catalog index: {index+gindex+1}) {CATALOG_REGISTRY_FEUPY.get_cls(catalog.tag)()}")


# In[ ]:




