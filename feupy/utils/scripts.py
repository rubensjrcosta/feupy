#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities to create scripts and command-line tools."""

import pickle
from gammapy.utils.scripts import make_path


# In[1]:


def pickling(object_instance, file_name):        
    """..."""
    with open(make_path(f"{file_name}.pkl"), "wb") as fp:  
        pickle.dump(object_instance, fp)
        
    return


# In[ ]:


def unpickling(file_name):        
    """..."""
    with open(make_path(f"{file_name}.pkl"), "rb") as fp:  
        return pickle.load(fp)

