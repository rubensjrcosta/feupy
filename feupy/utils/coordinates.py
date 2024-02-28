#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for coordinates."""

from astropy.coordinates import SkyCoord


# In[ ]:


def skcoord_to_dict(position: SkyCoord):
    return {
        'lon': position.ra,
        'lat': position.dec,
        'frame': position.frame.name,
    }


# In[1]:


def dict_to_skcoord(pos_dict: dict):
    return SkyCoord(pos_dict.lon, pos_dict.lat, frame=pos_dict.frame)


# In[ ]:




