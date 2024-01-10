#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.table import Table
from gammapy.utils.scripts import make_path

def get_table():
    file_name = "$PYTHONPATH/feupy/data/catalogs/1LHAASO_catalog.fits"
    return Table.read(make_path(file_name))


# # In[3]:


# table.info()

# table.to_pandas()


# In[ ]:




