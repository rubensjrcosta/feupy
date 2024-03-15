<<<<<<< Updated upstream
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pyflakes config.py


# In[1]:


=======
>>>>>>> Stashed changes
# Licensed under a 3-clause BSD style license - see LICENSE.rst
import html
import json
import logging
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import List
from astropy.coordinates import Angle
from astropy.time import Time
from astropy.units import Quantity
import yaml

<<<<<<< Updated upstream

# In[2]:


# from pydantic import BaseModel
=======
#from pydantic import BaseModel
#from pydantic.utils import deep_update
>>>>>>> Stashed changes
from pydantic.v1 import BaseModel
from pydantic.v1.utils import lenient_isinstance
from pydantic.utils import deep_update

from gammapy.makers import MapDatasetMaker
from gammapy.utils.scripts import make_path, read_yaml

from feupy.utils.types import (
    AngleType,
    EnergyType,
    QuantityType,
    TimeType,
    IrfType,
)

from feupy.utils.enum import(    
    ReductionTypeEnum,
    FrameEnum,
    RequiredHDUEnum,
    BackgroundMethodEnum,
    SafeMaskMethodsEnum,
    MapSelectionEnum,
)



__all__ = ["AnalysisConfig", "CounterpartsConfig", "SimulationCOnfig"]

CONFIG_PATH = Path(__file__).resolve().parent / "config"
DOCS_FILE = CONFIG_PATH / "docs.yaml"

log = logging.getLogger(__name__)






    
class RequiredAnalysisTypeEnum(str, Enum):
    gammapy = "gammapy"
    counterparts = "counterparts"
    cta_simulation = "cta_simulation"


    

class GammapyBaseConfig(BaseModel):
    class Config:
        validate_all = True
        validate_assignment = True
        extra = "forbid"
        json_encoders = {
            Angle: lambda v: f"{v.value} {v.unit}",
            Quantity: lambda v: f"{v.value} {v.unit}",
            Time: lambda v: f"{v.value}",
        }

    def _repr_html_(self):
        try:
            return self.to_html()
        except AttributeError:
            return f"<pre>{html.escape(str(self))}</pre>"


class SkyCoordConfig(GammapyBaseConfig):
    frame: FrameEnum = None
    lon: AngleType = None
    lat: AngleType = None


class EnergyAxisConfig(GammapyBaseConfig):
    min: EnergyType = None
    max: EnergyType = None
    nbins: int = None
    name: str = None

class SpatialCircleConfig(GammapyBaseConfig):
    frame: FrameEnum = None
    lon: AngleType = None
    lat: AngleType = None
    radius: AngleType = None


class EnergyRangeConfig(GammapyBaseConfig):
    min: EnergyType = None
    max: EnergyType = None


class TimeRangeConfig(GammapyBaseConfig):
    start: TimeType = None
    stop: TimeType = None


class FluxPointsConfig(GammapyBaseConfig):
    energy: EnergyAxisConfig = EnergyAxisConfig()
    source: str = "source"
    parameters: dict = {"selection_optional": "all"}


class LightCurveConfig(GammapyBaseConfig):
    time_intervals: TimeRangeConfig = TimeRangeConfig()
    energy_edges: EnergyAxisConfig = EnergyAxisConfig()
    source: str = "source"
    parameters: dict = {"selection_optional": "all"}


class FitConfig(GammapyBaseConfig):
    fit_range: EnergyRangeConfig = EnergyRangeConfig()


class ExcessMapConfig(GammapyBaseConfig):
    correlation_radius: AngleType = "0.1 deg"
    parameters: dict = {}
    energy_edges: EnergyAxisConfig = EnergyAxisConfig()


class BackgroundConfig(GammapyBaseConfig):
    method: BackgroundMethodEnum = None
    exclusion: Path = None
    parameters: dict = {}


class SafeMaskConfig(GammapyBaseConfig):
    methods: List[SafeMaskMethodsEnum] = [SafeMaskMethodsEnum.aeff_default]
    parameters: dict = {}


class EnergyAxesConfig(GammapyBaseConfig):
    energy: EnergyAxisConfig = EnergyAxisConfig(min="1 TeV", max="10 TeV", nbins=5)
    energy_true: EnergyAxisConfig = EnergyAxisConfig(
        min="0.5 TeV", max="20 TeV", nbins=16
    )


class SelectionConfig(GammapyBaseConfig):
    offset_max: AngleType = "2.5 deg"


class WidthConfig(GammapyBaseConfig):
    width: AngleType = "5 deg"
    height: AngleType = "5 deg"


class WcsConfig(GammapyBaseConfig):
    skydir: SkyCoordConfig = SkyCoordConfig()
    binsize: AngleType = "0.02 deg"
    width: WidthConfig = WidthConfig()
    binsize_irf: AngleType = "0.2 deg"


class GeomConfig(GammapyBaseConfig):
    wcs: WcsConfig = WcsConfig()
    selection: SelectionConfig = SelectionConfig()
    axes: EnergyAxesConfig = EnergyAxesConfig()

<<<<<<< Updated upstream

class DatasetsConfig(GammapyBaseConfig):
#     type: ReductionTypeEnum = ReductionTypeEnum.spectrum
#     stack: bool = True
    geom: GeomConfig = GeomConfig()
    selection: List[MapSelectionEnum] = MapDatasetMaker.available_selection
    use_region_center: bool = False
#     background: BackgroundConfig = BackgroundConfig()
    safe_mask: SafeMaskConfig = SafeMaskConfig()
#     on_region: SpatialCircleConfig = SpatialCircleConfig()
    containment_correction: bool = False
    acceptance: int = None
    acceptance_off: int = None
    alpha: float = None
=======
class DatasetsOnOffConfig(GammapyBaseConfig):
    acceptance: int = None
    acceptance_off: int = None

class DatasetsConfig(GammapyBaseConfig):
    type: ReductionTypeEnum = ReductionTypeEnum.spectrum
    stack: bool = True
    geom: GeomConfig = GeomConfig()
    map_selection: List[MapSelectionEnum] = MapDatasetMaker.available_selection
    background: BackgroundConfig = BackgroundConfig()
    safe_mask: SafeMaskConfig = SafeMaskConfig()
    on_region: SpatialCircleConfig = SpatialCircleConfig()
    containment_correction: bool = True
    use_region_center: bool = False

class ObservationsConfig(GammapyBaseConfig):
    datastore: Path = Path("$GAMMAPY_DATA/hess-dl3-dr1/")
    obs_ids: List[int] = []
    obs_file: Path = None
    obs_cone: SpatialCircleConfig = SpatialCircleConfig()
    obs_time: TimeRangeConfig = TimeRangeConfig()
    required_irf: List[RequiredHDUEnum] = ["aeff", "edisp", "psf", "bkg"]

class TargetConfig(GammapyBaseConfig):
    name: str = None
    position: SkyCoordConfig = SkyCoordConfig()
    model: dict = {}  
    model_fitted: dict = {}
        
# class PointingConfig(GammapyBaseConfig):
#     angle: AngleType = '0 deg'
        
class ROIConfig(GammapyBaseConfig):
    target: TargetConfig = TargetConfig()
    radius: AngleType = None
    catalogs: CatalogsTypeEnum = "all"
    dict_sep: dict = {} 
    leg_style: dict = {}

#class ROIConfig(GammapyBaseConfig):
#    target: TargetConfig = TargetConfig()
#    radius: AngleType = None
#     catalogs: CatalogsTypeEnum = "all"
#     dict_sep: dict = {} 
#     leg_style: dict = {}
                
class ObservationConfig(GammapyBaseConfig):
#     datastore: Path = Path("$GAMMAPY_DATA/hess-dl3-dr1/")
#     obs_ids: List[int] = []
#     obs_file: Path = None
    roi: ROIConfig = ROIConfig()
#     obs_time: QuantityType = "1 h"
#     required_irf: List[RequiredHDUEnum] = ["aeff", "edisp", "psf", "bkg"]
#     target: TargetConfig = TargetConfig()
    livetime: QuantityType = None
    offset: QuantityType = None
    pointing_angle: AngleType = '0 deg'
    
#     on_region_radius: AngleType = None
    
#     parameters: ObservationsParametersConfig = ObservationsParametersConfig()
    config_irf: IrfType = ['South', 'AverageAz', '20deg', '50h']
>>>>>>> Stashed changes
        
        
class SensitivityConfig(GammapyBaseConfig):
    gamma_min: int = None
    n_sigma: int = None
    bkg_syst_fraction: float = None

<<<<<<< Updated upstream
                
class ObservationsParametersConfig(GammapyBaseConfig):
    livetime: QuantityType = None
    offset: QuantityType = None
    on_region_radius: AngleType = None
    n_obs: int = None


class TargetConfig(GammapyBaseConfig):
    name: str = None
    position: SkyCoordConfig = SkyCoordConfig()
    model:  dict = {}  
        
class IrfsConfig(GammapyBaseConfig):
    opt: IrfType = ['South', 'AverageAz', '20deg', '50h']

class PointingConfig(GammapyBaseConfig):
    angle: AngleType = 0 * u.deg
        
class ObservationsConfig(GammapyBaseConfig):
    target: TargetConfig = TargetConfig()
    parameters: ObservationsParametersConfig = ObservationsParametersConfig()
    irfs: IrfsConfig = IrfsConfig()
    pointing: PointingConfig = PointingConfig()
        
class ROIConfig(GammapyBaseConfig):
    target: TargetConfig = TargetConfig()
    region_radius: AngleType = None
    
        
#     datastore: Path = Path("$GAMMAPY_DATA/hess-dl3-dr1/")
#     obs_ids: List[int] = []
#     obs_file: Path = None
#     obs_cone: SpatialCircleConfig = SpatialCircleConfig()
#     obs_time: TimeRangeConfig = TimeRangeConfig()
#     required_irf: List[RequiredHDUEnum] = ["aeff", "edisp", "psf", "bkg"]


=======
# class WStatisticsConfig(GammapyBaseConfig):
#     excess: float = None
#     ts: float = None
#     sqrt_ts: float = None

class StatisticsConfig(GammapyBaseConfig):
    alpha: float = None
    wstat: dict = {}
    fitted_parameters: dict = {}   

        
        
>>>>>>> Stashed changes
class LogConfig(GammapyBaseConfig):
    level: str = "info"
    filename: Path = None
    filemode: str = None
    format: str = None
    datefmt: str = None

class GeneralConfig(GammapyBaseConfig):
    log: LogConfig = LogConfig()
    outdir: str = "."
    n_jobs: int = 1
    datasets_file: Path = None
    models_file: Path = None
    type: RequiredAnalysisTypeEnum = "gammapy"
    n_obs: int = 10

<<<<<<< Updated upstream
        
=======
class GeneralCounterpartsConfig(GammapyBaseConfig):
    log: LogConfig = LogConfig()
    outdir: str = "."
    path_file: Path = None
        
        
class AnalysisConfig(GammapyBaseConfig):
    """Gammapy analysis configuration."""

    general: GeneralConfig = GeneralConfig()
    observation: ObservationConfig = ObservationConfig()
    observations: ObservationsConfig = ObservationsConfig()
    datasets: DatasetsConfig = DatasetsConfig()
    fit: FitConfig = FitConfig()
    flux_points: FluxPointsConfig = FluxPointsConfig()
    excess_map: ExcessMapConfig = ExcessMapConfig()
    light_curve: LightCurveConfig = LightCurveConfig()

    def __str__(self):
        """Display settings in pretty YAML format."""
        info = self.__class__.__name__ + "\n\n\t"
        data = self.to_yaml()
        data = data.replace("\n", "\n\t")
        info += data
        return info.expandtabs(tabsize=4)

    @classmethod
    def read(cls, path):
        """Read from YAML file."""
        config = read_yaml(path)
        return AnalysisConfig(**config)

    @classmethod
    def from_yaml(cls, config_str):
        """Create from YAML string."""
        settings = yaml.safe_load(config_str)
        return AnalysisConfig(**settings)

    def write(self, path, overwrite=False):
        """Write to YAML file."""
        path = make_path(path)
        if path.exists() and not overwrite:
            raise IOError(f"File exists already: {path}")
        path.write_text(self.to_yaml())

    def to_yaml(self):
        """Convert to YAML string."""
        # Here using `dict()` instead of `json()` would be more natural.
        # We should change this once pydantic adds support for custom encoders
        # to `dict()`. See https://github.com/samuelcolvin/pydantic/issues/1043
        config = json.loads(self.json())
        return yaml.dump(
            config, sort_keys=False, indent=4, width=80, default_flow_style=None
        )

    def set_logging(self):
        """Set logging config.

        Calls ``logging.basicConfig``, i.e. adjusts global logging state.
        """
        self.general.log.level = self.general.log.level.upper()
        logging.basicConfig(**self.general.log.dict())
        log.info("Setting logging config: {!r}".format(self.general.log.dict()))

    def update(self, config=None):
        """Update config with provided settings.

        Parameters
        ----------
        config : str or `AnalysisConfig` object, optional
            Configuration settings provided in dict() syntax. Default is None.
        """
        if isinstance(config, str):
            other = AnalysisConfig.from_yaml(config)
        elif isinstance(config, AnalysisConfig):
            other = config
        else:
            raise TypeError(f"Invalid type: {config}")

        config_new = deep_update(
            self.dict(exclude_defaults=True), other.dict(exclude_defaults=True)
        )
        return AnalysisConfig(**config_new)

    @staticmethod
    def _get_doc_sections():
        """Return dictionary with commented docs from docs file."""
        doc = defaultdict(str)
        with open(DOCS_FILE) as f:
            for line in filter(lambda line: not line.startswith("---"), f):
                line = line.strip("\n")
                if line.startswith("# Section: "):
                    keyword = line.replace("# Section: ", "")
                doc[keyword] += line + "\n"
        return doc

class CounterpartsConfig(GammapyBaseConfig):
    """Gammapy analysis configuration."""

    general: GeneralCounterpartsConfig = GeneralCounterpartsConfig()
    roi: ROIConfig = ROIConfig()
    energy_range: EnergyRangeConfig = EnergyRangeConfig()
#     irfs: IrfsConfig = IrfsConfig()
#     datasets: DatasetsConfig = DatasetsConfig()
#     sensitivity: SensitivityConfig = SensitivityConfig()
#     fit: FitConfig = FitConfig()
#     flux_points: FluxPointsConfig = FluxPointsConfig()
#     excess_map: ExcessMapConfig = ExcessMapConfig()
#     light_curve: LightCurveConfig = LightCurveConfig()

    def __str__(self):
        """Display settings in pretty YAML format."""
        info = self.__class__.__name__ + "\n\n\t"
        data = self.to_yaml()
        data = data.replace("\n", "\n\t")
        info += data
        return info.expandtabs(tabsize=4)

    @classmethod
    def read(cls, path):
        """Reads from YAML file."""
        config = read_yaml(path)
        return CounterpartsConfig(**config)


    @classmethod
    def from_yaml(cls, config_str):
        """Create from YAML string."""
        settings = yaml.safe_load(config_str)
        return CounterpartsConfig(**settings)


    def write(self, path = None, overwrite=False):
        """Write to YAML file."""
        if path is not None:
            path = make_path(path)
        else: path = make_path(f"{self.general.path_file}/counterparts_config.yaml")
            
        if path.exists() and not overwrite:
            raise IOError(f"File exists already: {path}")
        path.write_text(self.to_yaml())


    def to_yaml(self):
        """Convert to YAML string."""
        # Here using `dict()` instead of `json()` would be more natural.
        # We should change this once pydantic adds support for custom encoders
        # to `dict()`. See https://github.com/samuelcolvin/pydantic/issues/1043
        config = json.loads(self.json())
        return yaml.dump(
            config, sort_keys=False, indent=4, width=80, default_flow_style=None
        )

    def set_logging(self):
        """Set logging config.

        Calls ``logging.basicConfig``, i.e. adjusts global logging state.
        """
        self.general.log.level = self.general.log.level.upper()
        logging.basicConfig(**self.general.log.dict())
        log.info("Setting logging config: {!r}".format(self.general.log.dict()))


    def update(self, config=None):
        """Update config with provided settings.

        Parameters
        ----------
        config : string dict or `CounterpartsConfig` object
            Configuration settings provided in dict() syntax.
        """
        if isinstance(config, str):
            other = CounterpartsConfig.from_yaml(config)
        elif isinstance(config, CounterpartsConfig):
            other = config
        else:
            raise TypeError(f"Invalid type: {config}")

        config_new = deep_update(
            self.dict(exclude_defaults=True), other.dict(exclude_defaults=True)
        )
        return CounterpartsConfig(**config_new)


    @staticmethod
    def _get_doc_sections():
        """Returns dict with commented docs from docs file"""
        doc = defaultdict(str)
        with open(DOCS_FILE) as f:
            for line in filter(lambda line: not line.startswith("---"), f):
                line = line.strip("\n")
                if line.startswith("# Section: "):
                    keyword = line.replace("# Section: ", "")
                doc[keyword] += line + "\n"
        return doc    
>>>>>>> Stashed changes


# In[ ]:




class SimulationConfig(GammapyBaseConfig):
    """Gammapy analysis configuration."""

    general: GeneralConfig = GeneralConfig()
    observations: ObservationsConfig = ObservationsConfig()
#     irfs: IrfsConfig = IrfsConfig()
    datasets: DatasetsConfig = DatasetsConfig()
    sensitivity: SensitivityConfig = SensitivityConfig()
<<<<<<< Updated upstream
#     fit: FitConfig = FitConfig()
#     flux_points: FluxPointsConfig = FluxPointsConfig()
#     excess_map: ExcessMapConfig = ExcessMapConfig()
#     light_curve: LightCurveConfig = LightCurveConfig()
=======
    fit: FitConfig = FitConfig()
    flux_points: FluxPointsConfig = FluxPointsConfig()
>>>>>>> Stashed changes

    def __str__(self):
        """Display settings in pretty YAML format."""
        info = self.__class__.__name__ + "\n\n\t"
        data = self.to_yaml()
        data = data.replace("\n", "\n\t")
        info += data
        return info.expandtabs(tabsize=4)

    @classmethod
    def read(cls, path):
        """Reads from YAML file."""
        config = read_yaml(path)
        return SimulationConfig(**config)


    @classmethod
    def from_yaml(cls, config_str):
        """Create from YAML string."""
        settings = yaml.safe_load(config_str)
        return SimulationConfig(**settings)


    def write(self, path, overwrite=False):
        """Write to YAML file."""
        path = make_path(path)
        if path.exists() and not overwrite:
            raise IOError(f"File exists already: {path}")
        path.write_text(self.to_yaml())


    def to_yaml(self):
        """Convert to YAML string."""
        # Here using `dict()` instead of `json()` would be more natural.
        # We should change this once pydantic adds support for custom encoders
        # to `dict()`. See https://github.com/samuelcolvin/pydantic/issues/1043
        config = json.loads(self.json())
        return yaml.dump(
            config, sort_keys=False, indent=4, width=80, default_flow_style=None
        )

    def set_logging(self):
        """Set logging config.

        Calls ``logging.basicConfig``, i.e. adjusts global logging state.
        """
        self.general.log.level = self.general.log.level.upper()
        logging.basicConfig(**self.general.log.dict())
        log.info("Setting logging config: {!r}".format(self.general.log.dict()))


    def update(self, config=None):
        """Update config with provided settings.

        Parameters
        ----------
        config : string dict or `SimulationConfig` object
            Configuration settings provided in dict() syntax.
        """
        if isinstance(config, str):
            other = SimulationConfig.from_yaml(config)
        elif isinstance(config, SimulationConfig):
            other = config
        else:
            raise TypeError(f"Invalid type: {config}")

        config_new = deep_update(
            self.dict(exclude_defaults=True), other.dict(exclude_defaults=True)
        )
        return SimulationConfig(**config_new)


    @staticmethod
    def _get_doc_sections():
        """Returns dict with commented docs from docs file"""
        doc = defaultdict(str)
        with open(DOCS_FILE) as f:
            for line in filter(lambda line: not line.startswith("---"), f):
                line = line.strip("\n")
                if line.startswith("# Section: "):
                    keyword = line.replace("# Section: ", "")
                doc[keyword] += line + "\n"
        return doc


# In[ ]:


class CounterpartsConfig(GammapyBaseConfig):
    """Gammapy analysis configuration."""

    general: GeneralConfig = GeneralConfig()
    roi: ROIConfig = ROIConfig()
    energy_range: EnergyRangeConfig = EnergyRangeConfig()
#     irfs: IrfsConfig = IrfsConfig()
#     datasets: DatasetsConfig = DatasetsConfig()
#     sensitivity: SensitivityConfig = SensitivityConfig()
#     fit: FitConfig = FitConfig()
#     flux_points: FluxPointsConfig = FluxPointsConfig()
#     excess_map: ExcessMapConfig = ExcessMapConfig()
#     light_curve: LightCurveConfig = LightCurveConfig()

    def __str__(self):
        """Display settings in pretty YAML format."""
        info = self.__class__.__name__ + "\n\n\t"
        data = self.to_yaml()
        data = data.replace("\n", "\n\t")
        info += data
        return info.expandtabs(tabsize=4)

    @classmethod
    def read(cls, path):
        """Reads from YAML file."""
        config = read_yaml(path)
        return CounterpartsConfig(**config)


    @classmethod
    def from_yaml(cls, config_str):
        """Create from YAML string."""
        settings = yaml.safe_load(config_str)
        return CounterpartsConfig(**settings)


    def write(self, path, overwrite=False):
        """Write to YAML file."""
        path = make_path(path)
        if path.exists() and not overwrite:
            raise IOError(f"File exists already: {path}")
        path.write_text(self.to_yaml())


    def to_yaml(self):
        """Convert to YAML string."""
        # Here using `dict()` instead of `json()` would be more natural.
        # We should change this once pydantic adds support for custom encoders
        # to `dict()`. See https://github.com/samuelcolvin/pydantic/issues/1043
        config = json.loads(self.json())
        return yaml.dump(
            config, sort_keys=False, indent=4, width=80, default_flow_style=None
        )

    def set_logging(self):
        """Set logging config.

        Calls ``logging.basicConfig``, i.e. adjusts global logging state.
        """
        self.general.log.level = self.general.log.level.upper()
        logging.basicConfig(**self.general.log.dict())
        log.info("Setting logging config: {!r}".format(self.general.log.dict()))


    def update(self, config=None):
        """Update config with provided settings.

        Parameters
        ----------
        config : string dict or `CounterpartsConfig` object
            Configuration settings provided in dict() syntax.
        """
        if isinstance(config, str):
            other = CounterpartsConfig.from_yaml(config)
        elif isinstance(config, CounterpartsConfig):
            other = config
        else:
            raise TypeError(f"Invalid type: {config}")

        config_new = deep_update(
            self.dict(exclude_defaults=True), other.dict(exclude_defaults=True)
        )
        return CounterpartsConfig(**config_new)


    @staticmethod
    def _get_doc_sections():
        """Returns dict with commented docs from docs file"""
        doc = defaultdict(str)
        with open(DOCS_FILE) as f:
            for line in filter(lambda line: not line.startswith("---"), f):
                line = line.strip("\n")
                if line.startswith("# Section: "):
                    keyword = line.replace("# Section: ", "")
                doc[keyword] += line + "\n"
        return doc
    


# In[ ]:





# In[ ]:





# In[ ]:

