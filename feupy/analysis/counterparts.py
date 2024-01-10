#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst


# In[2]:


from feupy.roi import ROI
from feupy.target import Target

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.units import Quantity

from gammapy.utils.units import energy_unit_format

import logging
from pathlib import Path


# In[3]:


from feupy.utils.string_handling import name_to_txt


# In[4]:


__all__ = ["AnalysisConfig", "Analysis"]

# log = logging.getLogger(__name__)


# CONFIG_PATH = Path(__file__).resolve().parent / "config"
# DOCS_FILE = CONFIG_PATH / "docs.yaml"


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
        self.__target_name=target_name
        self.position=SkyCoord(pos_ra, pos_dec) 
        self.radius=radius
        if e_ref_min is not None:
            self.e_ref_min=Quantity(e_ref_min, "TeV")
        else: self.e_ref_min=e_ref_min
        if e_ref_max is not None:
            self.e_ref_max=Quantity(e_ref_max, "TeV")
        else: self.e_ref_max=e_ref_max
        self.energy_range=[self.e_ref_min, self.e_ref_max]
        self.target=Target(self.__target_name, self.position.ra, self.position.dec)
        self.roi=ROI(self.__target_name, self.position.ra, self.position.dec, self.radius)
#         self.catalogs_roi=get_catalogs(self.roi)
        
        # Actions to execute
        AnalysisConfig.all.append(self)
    
    @property
    # Property Decorator=Read-Only Attribute
    def info(self):
        info={}
        info["target_name"]=self.target_name
        info["position"]=self.position
        info["radius"]=self.radius
        info["energy_range"]=self.energy_range
        return info    
    
    @property
    def target_name(self):
        return self.__target_name

    #     @name.setter
    #     def name(self, value):
    #         if len(value) > 15:
    #             raise Exception("The name is too long!")
    #         else:
    #             self.__name=value

    @staticmethod
    def is_integer(num):
       # We will count out the floats that are point zero
       # For i.e: 5.0, 10.0
        if isinstance(num, float):
           # Count out the floats that are point zero
           return num.is_integer()
        elif isinstance(num, int):
            return True
        else: return False

    def __repr__(self):        
        ss = f"{self.__class__.__name__}("
        ss += f"target_name='{self.__target_name}', "
        ss += "pos_ra={}*u.Unit('{}'), ".format(self.position.ra.value, self.position.ra.unit)
        ss += "pos_dec={}*u.Unit('{}'), ".format(self.position.dec.value, self.position.dec.unit)
        ss += "radius={}*u.Unit('{}'), ".format(self.radius.value, self.radius.unit)
        if self.e_ref_min is None: ss += "e_ref_min=None, "
        else: ss += "e_ref_min=Quantity('{}'), ".format(energy_unit_format(self.e_ref_min).replace(' ', ''))
        if self.e_ref_max is None: ss += "e_ref_max=None)"
        else: ss += "e_ref_max=Quantity('{}'))".format(energy_unit_format(self.e_ref_max).replace(' ', ''))
        return ss


# In[ ]:





# In[6]:


# def cli_run_analysis(filename, out, overwrite):
#     """Performs automated data reduction process."""
#     config = AnalysisConfig.read(filename)
#     config.datasets.background.method = "reflected"
#     analysis = Analysis(config)
#     analysis.get_observations()
#     analysis.get_datasets()
#     analysis.datasets.write(out, overwrite=overwrite)
#     log.info(f"Datasets stored in {out} folder.")


# In[7]:


def test_analysis_confg():
    return AnalysisConfig(
        "LHAASO J1825-1326", 
        276.45* u.Unit('deg'), 
        -13.45* u.Unit('deg'),
        1* u.Unit('deg'),
        1* u.Unit('erg')
    )


# In[16]:


from feupy.scripts import gammapy_catalogs 

from feupy.catalog.pulsar.atnf import PSR_PARAMS
from feupy.catalog.pulsar.atnf import SourceCatalogATNF
from feupy.catalog.lhaaso import SourceCatalogPublishNatureLHAASO
from feupy.catalog.hawc import SourceCatalogExtraHAWC

from gammapy.datasets import FluxPointsDataset
from astropy.coordinates import SkyCoord
from astropy import units as u
from gammapy.modeling.models import SkyModel, Models
from gammapy.datasets import Datasets
import numpy as np
import pandas as pd 

from gammapy.estimators import FluxPoints

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
#         self.config.set_logging()
        self.catalogs = None
        self.datasets = None
        self.sources = None
        self.models = None
        self.dict_roi = None
        self.df_roi = None


#         self.fit = Fit()
        self.fit_result = None
        self.flux_points = None
        
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
            
    def get_catalogs(self):
        _catalogs = []
        catalogs_roi = []
        
        position = self.config.roi.position 
        radius = self.config.roi.radius 

        catalogs_gammapy = gammapy_catalogs.load_all_catalogs()
        _catalogs.extend(catalogs_gammapy)
        catalog_lhaaso = SourceCatalogPublishNatureLHAASO()
        _catalogs.append(catalog_lhaaso)
        catalog_hawc = SourceCatalogExtraHAWC()
        _catalogs.append(catalog_hawc)

        n_tot = len(_catalogs)
        for catalog in _catalogs:        
            # Selects only sources within the region of interest. 
            separation = position.separation(catalog.positions)

            mask_roi = separation < radius

            if len(catalog[mask_roi].table):
                catalogs_roi.append(catalog[mask_roi])
#                 n_roi += 1
            else:
                pass
#               catalogs_roi_no.append(f"{catalog.tag}: {catalog.description}")
        self.catalogs = catalogs_roi
    


    def get_datasets(self):
        '''
        Select a catalog subset (only sources within a region of interest)
        '''

        datasets = Datasets() # global datasets object
        models = Models()  # global models object
        sources = [] # global sources object
        n_sources = 0 # number of sources
        n_flux_points = 0 # number of flux points tables
    
        for catalog in self.catalogs:
            cat_tag = catalog.tag
            for source in catalog:
                n_sources+=1   
                source_name = source.name            
                try:
                    flux_points = source.flux_points

                    spectral_model = source.spectral_model()
                    spectral_model_tag = spectral_model.tag[1]

                    if cat_tag == 'gamma-cat' or cat_tag == 'hgps':
                        dataset_name = f'{source_name}: {cat_tag}'
                    else: dataset_name = source_name

                    file_name = name_to_txt(dataset_name)

                    model = SkyModel(
                        name = f"{file_name}_{spectral_model_tag}",
                        spectral_model = spectral_model,
                        datasets_names=dataset_name
                    )

                    dataset = FluxPointsDataset(
                        models = model,
                        data = flux_points, 
                        name =  dataset_name   
                    )

                    if any([self.config.e_ref_min !=  None, self.config.e_ref_max !=  None]):
                        dataset = self.cut_energy_flux_points(dataset)
            
                    n_flux_points+=1
                    models.append(model)  # Add the model to models()

                    sources.append(source)
                    datasets.append(dataset)

    #                 table = dataset.data.to_table(sed_type = cfg.sed_type_e2dnde, formatted = True)

    #                 # Writes the flux points table in the csv/fits format
    #                 utl.write_tables_csv(table, path_file, file_name)
    #                 utl.write_tables_fits(table, path_file, file_name)

                except Exception as error:
                    # By this way we can know about the type of error occurring
                    print(f'The error is: ({source_name}) {error}') 

        datasets.models = models
        # To save datasets and models
    #     utl.write_datasets_models(datasets,region_of_interest, datasets_name)

        print(f"Total number of Gammapy sources: {n_sources}")
        print(f"Total number of flux points tables: {n_flux_points}")
        
        self.sources = sources
        self.datasets = datasets
        self.models = models
        
    def cut_energy_flux_points(self, dataset):
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
            print(e_ref)
        if e_ref_max != None:
            mask_energy = np.zeros(len(flux_points.to_table()), dtype=bool)

            for m, e_ref in enumerate(flux_points.energy_ref):
                if e_ref <= e_ref_max:
                    mask_energy[m] = True

            flux_points_mask = flux_points.to_table()[mask_energy]
            flux_points = FluxPoints.from_table(flux_points_mask)     

        return FluxPointsDataset(models = models, data = flux_points, name = ds_name)
     
    def get_dict_roi(self):
        _dict_roi = {}

        roi_pos = self.config.roi.position 
        radius_roi = self.config.roi.radius 

        for index, source in enumerate(self.sources):
            source_pos = source.position
            sep = source.position.separation(roi_pos).deg
            _dict_roi[source.name] = {
                'position': source_pos,
                'separation':sep }

        self.dict_roi = _dict_roi
        
    def get_df_roi(self):
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
        df["dec.(deg)"] =df_dec
        df["Sep.(deg)"]= df_sep
        self.df_roi = df


# In[17]:


# # config.energy_range

# config=test_analysis_confg()

# analysis = Analysis(config)

# analysis.get_catalogs()
# analysis.catalogs

# analysis.get_datasets()

# print(analysis.datasets)

# analysis.sources

# analysis.get_dict_roi()

# analysis.dict_roi

# analysis.get_df_roi()

# analysis.df_roi


# In[18]:





# In[ ]:




