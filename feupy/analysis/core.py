#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst


# In[2]:


import numpy as np
import pandas as pd 

from feupy.roi import ROI
from feupy.target import Target
from feupy.utils.string_handling import name_to_txt
from feupy.utils.datasets import cut_energy_table_fp

from feupy.scripts import gammapy_catalogs 

from feupy.catalog.pulsar.atnf import SourceCatalogATNF
from feupy.catalog.lhaaso import SourceCatalogPublishNatureLHAASO
from feupy.catalog.hawc import SourceCatalogExtraHAWC

from feupy.analysis import CounterpartsAnalysisConfig

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.units import Quantity

from gammapy.utils.units import energy_unit_format

from gammapy.datasets import FluxPointsDataset
from gammapy.datasets import Datasets

from gammapy.modeling.models import SkyModel, Models

from gammapy.estimators import FluxPoints

from pathlib import Path


# In[4]:


__all__ = ["CounterpartsAnalysis"]


# In[3]:


class CounterpartsAnalysis:
    """Config-driven high level analysis interface.

    It is initialized by default with a set of configuration parameters and values declared in
    an internal high level interface model, though the user can also provide configuration
    parameters passed as a nested dictionary at the moment of instantiation. In that case these
    parameters will overwrite the default values of those present in the configuration file.

    Parameters
    ----------
    config : dict or `~gammapy.analysis.counterparts.CounterpartsAnalysisConfig`
        Configuration options following `CounterpartsAnalysisConfig` schema.
    """

    def __init__(self, config):
        self.config = config
        self.catalogs = None
        self.datasets = None
        self.sources = None
        self.models = None
        self.pulsars = None
        self.dict_roi = None
        self.df_roi = None

        
    @property
    def config(self):
        """Analysis configuration as an `~feupy.analysis.CounterpartsAnalysisConfig` object."""
        return self._config

    @config.setter
    def config(self, value):
        if isinstance(value, CounterpartsAnalysisConfig):
            self._config = value
        else:
            raise TypeError("config must be CounterpartsAnalysisConfig")
            
    def run(self):
        self._get_catalogs()
        self._get_datasets()
        self._get_dict_roi()
        self._get_df_roi()
        
    def _get_catalogs(self):
        _catalogs = []
        catalogs_roi = []
        
        position = self.config.roi.position 
        radius = self.config.roi.radius 

        _catalogs.extend(gammapy_catalogs.load_all_catalogs())
        _catalogs.append(SourceCatalogExtraHAWC())
        _catalogs.append(SourceCatalogPublishNatureLHAASO())
        _catalogs.append(SourceCatalogATNF())


        n_tot = len(_catalogs)
        for catalog in _catalogs:        
            # Selects only sources within the region of interest. 
            separation = position.separation(catalog.positions)

            mask_roi = separation < radius

            if len(catalog[mask_roi].table):
                catalogs_roi.append(catalog[mask_roi])
            else:
                pass
#               catalogs_roi_no.append(f"{catalog.tag}: {catalog.description}")
        self.catalogs = catalogs_roi
  
    def _get_datasets(self):
        """
        Select a catalog subset (only sources within a region of interest)
        """

        datasets = Datasets() # global datasets object
        models = Models()  # global models object
        sources = [] # global sources object
        pulsars = [] # global pulsars object
        n_sources = 0 # number of sources
        n_flux_points = 0 # number of flux points tables
        
        for catalog in self.catalogs:
            cat_tag = catalog.tag
            for source in catalog:
                n_sources += 1   
                source_name = source.name            
                if cat_tag == "ATNF":
                    pulsars.append(source)
                else:
                    try:
                        flux_points = source.flux_points

                        spectral_model = source.spectral_model()
                        spectral_model_tag = spectral_model.tag[1]

                        if cat_tag == 'gamma-cat' or cat_tag == 'hgps':
                            dataset_name = f'{source_name}: {cat_tag}'
                        else: dataset_name = source_name

                        file_name = name_to_txt(dataset_name)

                        model = SkyModel(
                            name=f"{file_name}_{spectral_model_tag}",
                            spectral_model=spectral_model,
                            datasets_names=dataset_name
                        )

                        dataset = FluxPointsDataset(
                            models=model,
                            data=flux_points, 
                            name=dataset_name   
                        )

                        if any([self.config.e_ref_min !=  None, self.config.e_ref_max !=  None]):
                            dataset = cut_energy_table_fp(dataset, self.config.e_ref_min, self.config.e_ref_max) 

                        n_flux_points += 1
                        models.append(model)  # Add the model to models()

                        sources.append(source)
                        datasets.append(dataset)

                    except Exception as error:
                        # By this way we can know about the type of error occurring
                        print(f'The error is: ({source_name}) {error}') 

        datasets.models = models
        
        self.pulsars = pulsars
        self.sources = sources
        self.datasets = datasets
        self.models = models
        
        print(f"Total number of gammapy sources: {n_sources-len(pulsars)}")
        print(f"Total number of flux points tables: {n_flux_points}")
        print(f"Total number of pulsars: {len(pulsars)}")
             
    def _get_dict_roi(self):
        _dict_roi = {}

        roi_pos = self.config.roi.position 
        radius_roi = self.config.roi.radius 

        _sources = self.sources.copy()
        _sources.extend(self.pulsars)
        for index, source in enumerate(_sources):
            source_pos = source.position
            sep = source.position.separation(roi_pos).deg
            if index < len(self.datasets):
                name = self.datasets[index].name
            else: name = source.name
            _dict_roi[name] = {
                'position': source_pos,
                'separation':sep
            }

        self.dict_roi = _dict_roi
        
    def _get_df_roi(self):
        _dict = self.dict_roi

        df = pd.DataFrame()
        df["Source name"] = _dict.keys()
        df_ra = []
        df_dec = []
        df_sep = []

        for index, name in enumerate(_dict.keys()):
            df_ra.append(_dict[name]["position"].ra.deg)
            df_dec.append(_dict[name]["position"].dec.deg)
            df_sep.append(_dict[name]["separation"])

        df["RA(deg)"] = df_ra
        df["dec.(deg)"] = df_dec
        df["Sep.(deg)"] = df_sep
        self.df_roi = df
        
    def _create_roi_name(self): 
        """ ... """
        ss = f"{self.config.target_name}"
        ss += "_roi_{:.2f}".format(self.config.radius).replace(' ', '')
        if self.config.e_ref_min is None: ss += ""
        else: ss += "_e_ref_min_{}".format(energy_unit_format(self.config.e_ref_min).replace(' ', ''))
        if self.config.e_ref_max is None: ss += ""
        else: ss += "_e_ref_max_{}".format(energy_unit_format(self.config.e_ref_max).replace(' ', ''))
        return ss
    
    def create_analysis_path(self): 
        """ ... """
        return Path(f"analysis_roi/{self._create_roi_name()}")

    def write_datasets_models(self, overwrite=True):
        """Write Datasets and Models to YAML file.

            Parameters
            ----------
            overwrite : bool, optional
                Overwrite existing file. Default is True.
            """

        path_file = Path(f"{self.create_analysis_path()}/datasets")
        path_file.mkdir(parents=True, exist_ok=True)
        self.datasets.write(filename=f"{path_file}/datasets.yaml", filename_models=f"{path_file}/models.yaml", overwrite=overwrite)
    
    def read_datasets_models(self):
        """Read Datasets and Models from YAML file."""

        path_file = Path(f"{self.create_analysis_path()}/datasets")
#         path_file.mkdir(parents=True, exist_ok=True)
        return Datasets.read(filename=f"{path_file}/datasets.yaml", filename_models=f"{path_file}/models.yaml")
#     def read_datasets_models():
#         path_file = Path(f"{PATH_ANALYSIS}/datasets")
#         path_file.mkdir(parents=True, exist_ok=True)
#         return Datasets.read(filename=f"{path_file}/datasets.yaml", filename_models=f"{path_file}/models.yaml")


# In[2]:


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


# In[ ]:





# In[7]:


def test_analysis_confg():
    return CounterpartsAnalysisConfig(
        "LHAASO J1825-1326", 
        276.45* u.Unit('deg'), 
        -13.45* u.Unit('deg'),
        1* u.Unit('deg'),
        1* u.Unit('erg')
    )


# In[10]:





# In[ ]:




