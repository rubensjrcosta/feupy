#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np

from gammapy.datasets import Datasets
from gammapy.datasets import FluxPointsDataset

from gammapy.estimators import FluxPoints

from gammapy.modeling.models import SkyModel

from feupy.utils.scripts import is_documented_by


# In[ ]:





# In[2]:


@is_documented_by([FluxPoints, FluxPointsDataset])
def flux_points_dataset_from_table(
    table,
    reference_model=None,
    sed_type=None,
    name=None,
    kwargs_fp={
        'format':'gadf-sed',
        'gti': None,
    },
    kwargs_ds={
        'mask_fit': None,
        'mask_safe': None,
        'meta_table': None,
    }
):
    flux_points = FluxPoints.from_table(
        table=table, 
        reference_model=reference_model, 
        sed_type=sed_type,
        **kwargs_fp,
    )
    models = SkyModel(spectral_model=reference_model, name=name)
    
    return FluxPointsDataset(
        models=models,
        data=flux_points, 
        name=name,
        **kwargs_ds,
    )


# In[3]:


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


# In[4]:


@is_documented_by(Datasets)
def write_datasets(datasets, filename=None, filename_models=None, overwrite=True):
    """Write Datasets and Models to YAML file.

        Parameters
        ----------
        overwrite : bool, optional
            Overwrite existing file. Default is True.
        """
    
    if filename is None:
        filename = "./datasets"
    else: filename.mkdir(parents=True, exist_ok=True)
    if filename_models:
        filename_models.mkdir(parents=True, exist_ok=True)
        
    datasets.write(filename=f"{filename}.yaml", filename_models=f"{filename_models}.yaml", overwrite=overwrite)


# In[5]:


def write_datasets(datasets, path_file=None, overwrite=True):
    """Write Datasets and Models to YAML file.

        Parameters
        ----------
        overwrite : bool, optional
            Overwrite existing file. Default is True.
        """
    
    if path_file is None:
        path_file = "."
    else: path_file.mkdir(parents=True, exist_ok=True)
    datasets.write(filename=f"{path_file}/datasets.yaml", filename_models=f"{path_file}/models.yaml", overwrite=overwrite)


# In[6]:


@is_documented_by([FluxPoints, FluxPointsDataset])
def read_datasets(path_file=None):
    """Read Datasets and Models from YAML file."""

    if path_file is None:
        path_file = "."
    else: path_file.mkdir(parents=True, exist_ok=True)
    return Datasets.read(filename=f"{path_file}/datasets.yaml", filename_models=f"{path_file}/models.yaml")


# In[ ]:


# # To save only the models
# models_3fhl.write("3fhl_models.yaml", overwrite=True)

# # To save datasets and models
# datasets.write(
#     filename="datasets-gc.yaml", filename_models="models_gc.yaml", overwrite=True
# )

# # To read only models
# models = Models.read("3fhl_models.yaml")
# print(models)

# # To read datasets with models
# datasets_read = Datasets.read("datasets-gc.yaml", filename_models="models_gc.yaml")
# print(datasets_read)

