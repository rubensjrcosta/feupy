#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst


# In[2]:


import logging
import numpy as np
import pandas as pd 

from feupy.roi import ROI
from feupy.target import Target
from feupy.utils.string_handling import name_to_txt
from feupy.scripts import gammapy_catalogs 

from feupy.catalog.pulsar.atnf import SourceCatalogATNF
from feupy.catalog.lhaaso import SourceCatalogPublishNatureLHAASO
from feupy.catalog.hawc import SourceCatalogExtraHAWC

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.units import Quantity

from gammapy.utils.units import energy_unit_format

from gammapy.datasets import FluxPointsDataset
from gammapy.datasets import Datasets

from gammapy.modeling.models import SkyModel, Models

from gammapy.estimators import FluxPoints

from pathlib import Path


# In[ ]:


PATH_ANALYSIS = Path("analysis")
PATH_ANALYSIS.mkdir(exist_ok=True)


# In[4]:


__all__ = ["AnalysisConfig", "Analysis"]


# In[5]:


# How to create a class:
class AnalysisConfig:
   # ADD others parameters
   
    # color="red" # The color of the flux ponts
    all=[]
    @u.quantity_input(pos_ra=u.deg, pos_dec=u.deg, radius=u.deg, e_ref_min=u.eV, e_ref_max=u.eV)
    def __init__(self, target_name: str, pos_ra, pos_dec, radius, e_ref_min=None, e_ref_max=None, catalogs_roi=None):
       # Run validations to the received arguments
        assert  0 <= pos_ra.value <= 360, f"Right Ascension {pos_ra} is not in the range: (0,360) deg!"
        assert -90 <= pos_dec.value <= 90, f"Declination {pos_dec} is not in the range: (-90,90) deg!"

        # Assign to self object
        self.__target_name = target_name
        self.position = SkyCoord(pos_ra, pos_dec) 
        self.radius = radius
        if e_ref_min is not None:
            self.e_ref_min = Quantity(e_ref_min, "TeV")
        else: self.e_ref_min = e_ref_min
        if e_ref_max is not None:
            self.e_ref_max = Quantity(e_ref_max, "TeV")
        else: self.e_ref_max = e_ref_max
        self.energy_range = [self.e_ref_min, self.e_ref_max]
        self.target = Target(self.__target_name, self.position.ra, self.position.dec)
        self.roi = ROI(self.__target_name, self.position.ra, self.position.dec, self.radius)
        
        # Actions to execute
        AnalysisConfig.all.append(self)
    
    @property
    # Property Decorator=Read-Only Attribute
    def info(self):
        info={}
        info["target_name"] = self.target_name
        info["position"] = self.position
        info["radius"] = self.radius
        info["energy_range"] = self.energy_range
        return info    
    
    @property
    def target_name(self):
        return self.__target_name

    def __repr__(self):
        ss = f"{self.__class__.__name__}("
        ss += f"target_name={self.__target_name}, "
        ss += "pos_ra=Quantity('{:.2f}'), ".format(self.position.ra).replace(' ', '')
        ss += "pos_dec=Quantity('{:.2f}'), ".format(self.position.dec).replace(' ', '')
        ss += "radius=Quantity('{:.2f}'), ".format(self.radius).replace(' ', '')
        if self.e_ref_min is None: ss += "e_ref_min=None, "
        else: ss += "e_ref_min=Quantity('{}'), ".format(energy_unit_format(self.e_ref_min).replace(' ', ''))
        if self.e_ref_max is None: ss += "e_ref_max=None)"
        else: ss += "e_ref_max=Quantity('{}'))".format(energy_unit_format(self.e_ref_max).replace(' ', ''))
        return ss 


# In[2]:


class Analysis:
    """Config-driven high level analysis interface.

    It is initialized by default with a set of configuration parameters and values declared in
    an internal high level interface model, though the user can also provide configuration
    parameters passed as a nested dictionary at the moment of instantiation. In that case these
    parameters will overwrite the default values of those present in the configuration file.

    Parameters
    ----------
    config : dict or `~gammapy.analysis.AnalysisConfig`
        Configuration options following `AnalysisConfig` schema.
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
        """Analysis configuration as an `~feupy.analysis.counterparts.AnalysisConfig` object."""
        return self._config

    @config.setter
    def config(self, value):
        if isinstance(value, AnalysisConfig):
            self._config = value
        else:
            raise TypeError("config must be AnalysisConfig.")
            
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
                            dataset = self._cut_energy_flux_points(dataset)

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
        
    def _cut_energy_flux_points(self, dataset):
        _datasets = Datasets()
        e_ref_min = self.config.e_ref_min
        e_ref_max = self.config.e_ref_max

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
        
    def write_datasets_models(self, overwrite=True):
        """Write datasets and Models to YAML file.

            Parameters
            ----------
            overwrite : bool, optional
                Overwrite existing file. Default is True.
            """
        path_file = Path(f"{PATH_ANALYSIS}/datasets")
        path_file.mkdir(parents=True, exist_ok=True)
        self.datasets.write(filename=f"{path_file}/datasets.yaml", filename_models=f"{path_file}/models.yaml", overwrite=overwrite)
        
        
#     def read_datasets_models():
#         path_file = Path(f"{PATH_ANALYSIS}/datasets")
#         path_file.mkdir(parents=True, exist_ok=True)
#         return Datasets.read(filename=f"{path_file}/datasets.yaml", filename_models=f"{path_file}/models.yaml")


# In[9]:


# analysis_confg = AnalysisConfig(
#     "LHAASO J1825-1326", 
#     276.45* u.Unit('deg'), 
#     -13.45* u.Unit('deg'),
#     1* u.Unit('deg'),
#     1* u.Unit('erg')
# )
# analysis = Analysis(analysis_confg)
# analysis.run()


# In[ ]:





# In[7]:


def test_analysis_confg():
    return AnalysisConfig(
        "LHAASO J1825-1326", 
        276.45* u.Unit('deg'), 
        -13.45* u.Unit('deg'),
        1* u.Unit('deg'),
        1* u.Unit('erg')
    )


# In[10]:





# In[ ]:




