#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Datasets."""


# In[1]:


from gammapy.datasets import SpectrumDataset
from gammapy.datasets import SpectrumDatasetOnOff


# In[2]:


__all__ = [
    "create_spectrum_dataset_empty",
    "create_spectrum_dataset_onoff"
]


# In[ ]:


def create_spectrum_dataset_empty(geom, energy_axis_true, name="obs-0"):
    """Create a MapDataset object with zero filled maps."""
    return SpectrumDataset.create(
        geom=geom, 
        energy_axis_true=energy_axis_true,
        name=name,
    )


# In[ ]:


def create_spectrum_dataset_onoff(dataset, acceptance, acceptance_off):
    """Create a SpectrumDatasetOnOff from a `SpectrumDataset` dataset.""" 
    return SpectrumDatasetOnOff.from_spectrum_dataset(
        dataset=dataset, 
        acceptance=acceptance, 
        acceptance_off=acceptance_off,
    )


# In[ ]:




