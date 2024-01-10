#!/usr/bin/env python
# coding: utf-8

# In[9]:


from gammapy.catalog import CATALOG_REGISTRY 


# In[15]:


def catalogs_registry():
    """
    """
    return CATALOG_REGISTRY


# In[10]:


def load_catalog(tag=None):
    """
    """
    return  CATALOG_REGISTRY.get_cls(tag)()


# In[11]:


def load_all_catalogs():
    """
    load all available source catalogs in gammapy.catalog package into a list
    
    >>> load_catalogs_from_gammapy()
    
    Source catalogs in Gammapy: 8

    (catalog index: 0) SourceCatalogGammaCat:
        name: gamma-cat
        description: An open catalog of gamma-ray sources
        sources: 162

    (catalog index: 1) SourceCatalogHGPS:
        name: hgps
        description: H.E.S.S. Galactic plane survey (HGPS) source catalog
        sources: 78

    (catalog index: 2) SourceCatalog2HWC:
        name: 2hwc
        description: 2HWC catalog from the HAWC observatory
        sources: 40

    (catalog index: 3) SourceCatalog3FGL:
        name: 3fgl
        description: LAT 4-year point source catalog
        sources: 3034

    (catalog index: 4) SourceCatalog4FGL:
        name: 4fgl
        description: LAT 8-year point source catalog
        sources: 6659

    (catalog index: 5) SourceCatalog2FHL:
        name: 2fhl
        description: LAT second high-energy source catalog
        sources: 360

    (catalog index: 6) SourceCatalog3FHL:
        name: 3fhl
        description: LAT third high-energy source catalog
        sources: 1556

    (catalog index: 7) SourceCatalog3HWC:
        name: 3hwc
        description: 3HWC catalog from the HAWC observatory
        sources: 65
    """
    source_catalogs = []
    for index, catalog in enumerate(CATALOG_REGISTRY):
        #  FITS files are loaded
        catalog_cls = CATALOG_REGISTRY.get_cls(catalog.tag)()
        source_catalogs.append(catalog_cls)
    return source_catalogs


# In[14]:


def catalogs_info():
    """
    """
    print (f"Source catalogs in Gammapy: {len(CATALOG_REGISTRY)}\n")
    for index, catalog in enumerate(CATALOG_REGISTRY):
        print(f"(catalog index: {index}) {CATALOG_REGISTRY.get_cls(catalog.tag)()}")


# In[ ]:





# In[ ]:




