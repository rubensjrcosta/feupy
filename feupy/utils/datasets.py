#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gammapy.estimators import FluxPoints

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




