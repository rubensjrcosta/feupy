#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

from gammapy.datasets import Datasets
from gammapy.datasets import FluxPointsDataset

from gammapy.estimators import FluxPoints


# In[1]:


def ds_fp_from_table_fp(table, sky_model, source_name, sed_type = "e2dnde"):
    '''Returns the flux points dataset from the flux points table 
    
    >>> ds_fp_from_table_fp(table, sky_model, sed_type)
    ds_fp
    '''
    flux_points = FluxPoints.from_table(table=table, reference_model=sky_model, sed_type=sed_type)
    
    return FluxPointsDataset(
        models=sky_model,
        data=flux_points, 
        name=source_name
    )


# In[ ]:


def cut_energy_table_fp(dataset, e_ref_min=None, e_ref_max=None):
    _datasets = Datasets()

    flux_points = dataset.data
    models = dataset.models[0]      
    ds_name = dataset.name

    if e_ref_min != None:
        mask_energy = np.zeros(len(flux_points.to_table()), dtype=bool)

        for m, e_ref in enumerate(flux_points.energy_ref):
            if e_ref >= e_ref_min:
                mask_energy[m] = True

        flux_points_mask = flux_points.to_table()[mask_energy]
        flux_points = FluxPoints.from_table(flux_points_mask)

    if e_ref_max != None:
        mask_energy = np.zeros(len(flux_points.to_table()), dtype=bool)

        for m, e_ref in enumerate(flux_points.energy_ref):
            if e_ref <= e_ref_max:
                mask_energy[m] = True

        flux_points_mask = flux_points.to_table()[mask_energy]
        flux_points = FluxPoints.from_table(flux_points_mask)     

    return FluxPointsDataset(models=models, data=flux_points, name=ds_name)

