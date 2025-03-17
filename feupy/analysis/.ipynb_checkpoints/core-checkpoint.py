# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Session class driving the high level interface API"""
import logging

from astropy.table import Table
import astropy.units as u


from gammapy.datasets import Datasets, FluxPointsDataset
from gammapy.utils.scripts import make_path
from gammapy.modeling.models import (
    SkyModel, 
    Models,
    DatasetModels, 
)

from feupy.catalog.utils import load_catalogs
from feupy.utils.coordinates import convert_pos_config_to_skycoord
from feupy.utils.datasets import cut_energy_flux_points_datasets
from feupy.roi.config import AnalysisRoiConfig
from feupy.sources import Sources
from feupy.catalog.hawc import get_flux_points_3hwc, get_flux_points_2hwc
from feupy.catalog.utils import get_catalog_tag
from feupy.catalog.veritas import generate_unique_name
 

__all__ = ["AnalysisRoi"]


log = logging.getLogger(__name__)

class AnalysisRoi:
    """Config-driven high level simulation interface.

    It is initialized by default with a set of configuration parameters and values declared in
    an internal high level interface model, though the user can also provide configuration
    parameters passed as a nested dictionary at the moment of instantiation. In that case these
    parameters will overwrite the default values of those present in the configuration file.

    Parameters
    ----------
    config : dict or `AnalysisRoiConfig`
        Configuration options following `AnalysisRoiConfig` schema
    """

    def __init__(self, config):
        self.config = config
        self.config.set_logging()
        self.datasets = None
        self.sources = None
        self._has_fp = None
        self.catalog = None
        
    @property
    def config(self):
        """Simulation configuration (`AnalysisRoiConfig`)"""
        return self._config

    @config.setter
    def config(self, value):
        if isinstance(value, dict):
            self._config = AnalysisRoiConfig(**value)
        elif isinstance(value, AnalysisRoiConfig):
            self._config = value
        else:
            raise TypeError("config must be dict or AnalysisRoiConfig.")

    @property
    def models(self):
        if not self.datasets:
            raise RuntimeError("No datasets defined. Impossible to set models.")
        return self.datasets.models

    @models.setter
    def models(self, models):
        self.set_models(models, extend=False)
    
    
    def _get_catalogs(self):
        """Load source catalogs and filter sources within the ROI."""
        catalogs = []
        config_settings = self.config
        all_catalogs = load_catalogs()
        position = convert_pos_config_to_skycoord(config_settings.roi.position)
        radius = config_settings.roi.radius

        for catalog in all_catalogs:
            separation = position.separation(catalog.positions)
            mask_roi = separation < radius

            if len(catalog[mask_roi].table):
                catalogs.append(catalog[mask_roi])
        return catalogs

    def _get_sources(self):
        """Retrieve sources from catalogs within the ROI."""
        sources = []
        for catalog in self._get_catalogs():
            sources.extend(catalog)
        self.sources =  Sources(sources)

    def _get_catalog_roi(self):
        """
        Compute separations of sources from the target position and return an Astropy Table.

        Returns
        -------
        result_table : `~astropy.table.Table`
            Table with columns for source name, catalog, RA, Dec, and separation from target.
        """
        names, ras, decs, separations, catalogs, has_fp = [], [], [], [], [], []
        sources = self.sources
        config_settings = self.config
        target_position = convert_pos_config_to_skycoord(config_settings.roi.position)

        for source in sources:
            source_name = source.name
            source_pos = source.position
            ra, dec = source_pos.ra.deg, source_pos.dec.deg
            catalog = get_catalog_tag(source)
            sep = source_pos.separation(target_position).deg

            names.append(source_name)
            ras.append(ra)
            decs.append(dec)
            catalogs.append(catalog)
            separations.append(sep)
            has_fp.append(self._has_fp[source.name])
            
        result_table = Table()
        result_table['index'] =  [_ for _ in range(0, (len(names)))]
        result_table['source_name'] = names
        result_table['source_label'] = sources.labels
        result_table['has_fp'] = has_fp
        result_table['catalog'] = catalogs
        
        result_table['ra'] = ras * u.deg
        result_table['dec'] = decs * u.deg
        result_table['separation'] = separations * u.deg

        for column in result_table.colnames:
            if column.startswith(("ra", "dec", "sep")):
                result_table[column].format = ".3f"
                
        # Add meta (columns descriptions)
        result_table.meta['description'] = {
            'index': "Unique identifier for each source.",
            'source_name': "Name of the source as listed in the catalog.",
            'source_label': "User-defined label for the source, often used for plotting.",
            'has_fp': "Boolean flag indicating if flux points are available (True/False).",
            'catalog': "Catalog from which the source originates.",
            'ra': "Right Ascension (RA) of the source in degrees, formatted to 3 decimal places.",
            'dec': "Declination (Dec) of the source in degrees, formatted to 3 decimal places.",
            'separation': "Angular separation from a reference position in degrees, formatted to 3 decimal places."
        }
        self.catalog = result_table      


    def run(self):
        """Run the ROI analysis, initializing datasets."""
        self._get_sources()
        self._get_flux_points_datasets()
        self._get_catalog_roi()

    def _get_flux_points_datasets(self):
        """Generate FluxPointsDataset objects from sources within the ROI."""
        models = Models()
        datasets = Datasets()
        dict_fp = {}
        sources = self.sources
        log.info("Generating FluxPointsDataset.")

        for index, source in enumerate(sources):
            label = sources.labels[index]
            if get_catalog_tag(source) == "psrcat":
                dict_fp[source.name] = False
                log.info("Flux points unavailable for ATNF Pulsar Catalog.")
                continue     
#             if get_catalog_tag(source) == "vtscat":
#                 dict_fp[source.name] = False
#                 log.info("VTSCat catalog from the VTSCat observatory")
#                 continue   
                
            try:
                which = None
                
                if get_catalog_tag(source) == "3hwc":
                    spectral_model = self._create_spectral_model(source, which)
                    flux_points = get_flux_points_3hwc(source)
                    dict_fp[source.name] = True
                    self._create_flux_points_dataset(label, models, datasets,  spectral_model, flux_points)
                                                     
                elif get_catalog_tag(source) == "2hwc":
                    for model_type in ['point', 'extended']:
                        spectral_model = self._create_spectral_model(source, model_type)
                        flux_points = get_flux_points_2hwc(source, model_type)
                        dict_fp[source.name] = True
                        self._create_flux_points_dataset(f"{label} ({model_type})", models, datasets,  spectral_model, flux_points)
                         
                elif get_catalog_tag(source) == "1LHAASO":
                    for model_type in ['KM2A', 'WCDA']:
                        spectral_model = self._create_spectral_model(source, model_type)
                        flux_points = source.flux_points(model_type)
                        dict_fp[source.name] = True
                        self._create_flux_points_dataset(f"{label} ({model_type})", models, datasets,  spectral_model, flux_points)

                        
                elif get_catalog_tag(source) in ['gamma-cat', 'hgps', '3fgl', '4fgl', '2fhl', '3fhl', 'LHAASO', 'extraHAWC', 'veritas-2018apj']:
                        spectral_model = source.spectral_model()
                        flux_points = source.flux_points
                        dict_fp[source.name] = True
                        self._create_flux_points_dataset(label, models, datasets,  spectral_model, flux_points)
                
                elif get_catalog_tag(source) == "vtscat":
                    unique_names = [source.name]
                    for flux_points in source.flux_points():
                        reference_id = flux_points.meta['reference_id']
                        model_name = generate_unique_name(source.name, reference_id, unique_names)
                        unique_names.append(model_name)
                        model = flux_points.reference_model
                        dict_fp[source.name] = True
                        spectral_model = model.spectral_model
                        label = model_name
                        self._create_flux_points_dataset(label, models, datasets,  spectral_model, flux_points)
                    
            except Exception as error:
                dict_fp[source.name] = False
                log.info(f"Unable to create dataset for {source.name}. Error: {error}")

        self.datasets = datasets
        self.datasets.models = models
        self._has_fp = dict_fp

    @staticmethod
    def _create_spectral_model(source, which=None):
        """Helper function to create a FluxPointsDataset for a source."""
        return source.spectral_model(which=which) if which else source.spectral_model()


    def _create_flux_points_dataset(self,  name, models, datasets,  spectral_model, flux_points):
        """Helper function to create a FluxPointsDataset for a source."""
        
        e_ref_min = self.config.energy_range.min
        e_ref_max = self.config.energy_range.max
        log.info("Getting FluxPointsDataset.")
        
        model = SkyModel(
        name=name,
        spectral_model=spectral_model,
        datasets_names=name
        )
        dataset = FluxPointsDataset(
        models=model,
        data=flux_points,
        name=name
        )
        
        if any([e_ref_min !=  None, e_ref_max !=  None]):
            dataset = cut_energy_flux_points_datasets(
            dataset, 
            e_ref_min, 
            e_ref_max
            ) 
                    

        models.append(model)
        datasets.append(dataset)
        log.info(f"Dataset created: {dataset}")

    def set_models(self, models, extend=True):
        """Set models on datasets.
        Adds `FoVBackgroundModel` if not present already

        Parameters
        ----------
        models : `~gammapy.modeling.models.Models` or str
            Models object or YAML models string
        extend : bool
            Extend the exiting models on the datasets or replace them.
        """
        if not self.datasets or len(self.datasets) == 0:
            raise RuntimeError("Missing datasets")

        log.info("Reading model.")
        if isinstance(models, str):
            models = Models.from_yaml(models)
        elif isinstance(models, Models):
            pass
        elif isinstance(models, DatasetModels) or isinstance(models, list):
            models = Models(models)
        else:
            raise TypeError(f"Invalid type: {models!r}")

        if extend:
            models.extend(self.datasets.models)

        self.datasets.models = models


        log.info(models)

    def read_models(self, path, extend=True):
        """Read models from YAML file.

        Parameters
        ----------
        path : str
        Path to the model file.
        extend : bool, optional
        Extend the exiting models on the datasets or replace them.
        Default is True.
        """
        path = make_path(path)
        models = Models.read(path)
        self.set_models(models, extend=extend)
        log.info(f"Models loaded from {path}.")
 
    def read_sources(self, path):
        """Read sources from YAML file.

        Parameters
        ----------
        path : str
        Path to the model file.
        extend : bool, optional
        Extend the exiting sources on the datasets or replace them.
        Default is True.
        """
        path = make_path(path)
        _sources = Sources()
        _sources.read(path)
        self.sources = Sources(_sources)
#         self.set_sources(sources, extend=extend)
        log.info(f"sources loaded from {path}.")
    
    def write_sources(self, overwrite=True):
        """Write sources to YAML file.

        File name is taken from the configuration file.
        """
        filename_sources = self.config.general.sources_file
        if filename_sources is not None:
            self.sources.write(
                filename_sources, overwrite=overwrite)
            log.info(f"sources loaded from {filename_sources}.")
        else:
            raise RuntimeError("Missing sources_file in config.general")
            
    def write_models(self, overwrite=True, write_covariance=True):
        """Write models to YAML file.

        File name is taken from the configuration file.
        """
        filename_models = self.config.general.models_file
        if filename_models is not None:
            self.models.write(
                filename_models, overwrite=overwrite, write_covariance=write_covariance
            )
            log.info(f"Models loaded from {filename_models}.")
        else:
            raise RuntimeError("Missing models_file in config.general")

    def read_datasets(self):
        """Read datasets from YAML file.

        File names are taken from the configuration file.
        """
        filename = self.config.general.datasets_file
        filename_models = self.config.general.models_file
        if filename is not None:
            self.datasets = Datasets.read(filename)
            log.info(f"Datasets loaded from {filename}.")
            if filename_models is not None:
                self.read_models(filename_models, extend=False)
        else:
            raise RuntimeError("Missing datasets_file in config.general")

    def write_datasets(self, overwrite=True, write_covariance=True):
        """Write datasets to YAML file.

        File names are taken from the configuration file.

        Parameters
        ----------
        overwrite : bool, optional
            Overwrite existing file. Default is True.
        write_covariance : bool, optional
            Save covariance or not. Default is True.
        """
        filename = self.config.general.datasets_file
        filename_models = self.config.general.models_file
        if filename is not None:
            self.datasets.write(
                filename,
                filename_models,
                overwrite=overwrite,
                write_covariance=write_covariance,
            )
            log.info(f"Datasets stored to {filename}.")
            log.info(f"Datasets stored to {filename_models}.")
        else:
            raise RuntimeError("Missing datasets_file in config.general")
