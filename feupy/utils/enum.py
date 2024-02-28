#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for enum validation."""


# In[1]:


from enum import Enum


# In[4]:


__all__ = [
    "ReductionTypeEnum",
    "FrameEnum",
    "RequiredHDUEnum",
    "BackgroundMethodEnum",
    "SafeMaskMethodsEnum",
    "MapSelectionEnum",
]


# In[3]:


class ReductionTypeEnum(str, Enum):
    spectrum = "1d"
    cube = "3d"


class FrameEnum(str, Enum):
    icrs = "icrs"
    galactic = "galactic"


class RequiredHDUEnum(str, Enum):
    events = "events"
    gti = "gti"
    aeff = "aeff"
    bkg = "bkg"
    edisp = "edisp"
    psf = "psf"
    rad_max = "rad_max"


class BackgroundMethodEnum(str, Enum):
    reflected = "reflected"
    fov = "fov_background"
    ring = "ring"


class SafeMaskMethodsEnum(str, Enum):
    aeff_default = "aeff-default"
    aeff_max = "aeff-max"
    edisp_bias = "edisp-bias"
    offset_max = "offset-max"
    bkg_peak = "bkg-peak"


class MapSelectionEnum(str, Enum):
    counts = "counts"
    exposure = "exposure"
    background = "background"
    psf = "psf"
    edisp = "edisp"


# In[ ]:




