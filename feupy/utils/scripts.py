#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities to create scripts and command-line tools."""

import pickle
from gammapy.utils.scripts import make_path
from astropy.coordinates import SkyCoord


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


# In[ ]:


def is_documented_by(original):
    def wrapper(target):
        ss = '*** Docstring of internal function\class ***\n'
        if isinstance(original, list):
            for _original in original:
                ss += f"{_original.__qualname__}:\n"
                ss += f"{_original.__doc__}\n"
        else:
            ss += f"{original.__doc__}\n"
        if target.__doc__:
            ss += f'\n*** Docstring of {target.__qualname__} ***\n'
            ss += f"{target.__doc__}"
        target.__doc__ = ss
        return target
    return wrapper


# In[ ]:





# In[ ]:




