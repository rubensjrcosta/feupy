# Licensed under a 3-clause BSD style license - see LICENSE
"""Configuration classes for ROI and CTAO analyses."""

import json
import logging
from collections import defaultdict
from pathlib import Path

import yaml

from gammapy.analysis.config import (
    EnergyRangeConfig,
    FitConfig,
    FluxPointsConfig,
    GammapyBaseConfig,
    GeneralConfig,
    SpatialCircleConfig,
    deep_update,
)
from gammapy.utils.scripts import make_path, read_yaml

from feupy.utils.config import (
    DatasetsConfig,
    ObservationConfig,
    SensitivityConfig,
    StatisticsConfig,
)

__all__ = ["ROIAnalysisConfig", "CTAOAnalysisConfig"]

CONFIG_PATH = Path(__file__).resolve().parent / "config"
DOCS_FILE = CONFIG_PATH / "docs.yaml"

log = logging.getLogger(__name__)


class ROIAnalysisConfig(GammapyBaseConfig):
    """Configuration for region-of-interest analyses."""

    general: GeneralConfig = GeneralConfig()
    roi: SpatialCircleConfig = SpatialCircleConfig()
    energy_range: EnergyRangeConfig = EnergyRangeConfig()

    def __str__(self):
        """Return the configuration in a readable YAML representation."""
        info = self.__class__.__name__ + "\n\n\t"
        data = self.to_yaml().replace("\n", "\n\t")
        info += data
        return info.expandtabs(tabsize=4)

    @classmethod
    def read(cls, path):
        """Read the configuration from a YAML file."""
        config = read_yaml(path)
        return cls(**config)

    @classmethod
    def from_yaml(cls, config_str):
        """Create a configuration from a YAML string."""
        settings = yaml.safe_load(config_str)
        return cls(**settings)

    def write(self, path, overwrite=False):
        """Write the configuration to a YAML file."""
        path = make_path(path)

        if path.exists() and not overwrite:
            raise OSError(f"File exists already: {path}")

        path.write_text(self.to_yaml())

    def to_yaml(self):
        """Convert the configuration to a YAML string."""
        data = json.loads(self.model_dump_json())
        return yaml.dump(
            data, sort_keys=False, indent=4, width=80, default_flow_style=None
        )

    def set_logging(self):
        """Configure logging using the settings in ``general.log``."""
        self.general.log.level = self.general.log.level.upper()
        logging.basicConfig(**self.general.log.model_dump())
        log.info("Setting logging config: %r", self.general.log.model_dump())

    def update(self, config=None):
        """Update the configuration with the provided settings.

        Parameters
        ----------
        config : str or `ROIAnalysisConfig`
            Configuration settings. A string is interpreted as YAML.

        Returns
        -------
        config : `ROIAnalysisConfig`
            Updated configuration.
        """
        if isinstance(config, str):
            other = ROIAnalysisConfig.from_yaml(config)
        elif isinstance(config, ROIAnalysisConfig):
            other = config
        else:
            raise TypeError(f"Invalid type: {config}")

        config_new = deep_update(
            self.model_dump(exclude_defaults=True),
            other.model_dump(exclude_defaults=True),
        )
        return ROIAnalysisConfig(**config_new)

    @staticmethod
    def _get_doc_sections():
        """Return documentation sections defined in the docs YAML file."""
        doc = defaultdict(str)
        with open(DOCS_FILE) as file:
            for line in filter(lambda line: not line.startswith("---"), file):
                line = line.strip("\n")
                if line.startswith("# Section: "):
                    keyword = line.replace("# Section: ", "")
                doc[keyword] += line + "\n"
        return doc


class CTAOAnalysisConfig(GammapyBaseConfig):
    """Configuration for CTAO analyses."""

    general: GeneralConfig = GeneralConfig()
    observation: ObservationConfig = ObservationConfig()
    datasets: DatasetsConfig = DatasetsConfig()
    statistics: StatisticsConfig = StatisticsConfig()
    fit: FitConfig = FitConfig()
    flux_points: FluxPointsConfig = FluxPointsConfig()
    sensitivity: SensitivityConfig = SensitivityConfig()

    def __str__(self):
        """Return the configuration in a readable YAML representation."""
        info = self.__class__.__name__ + "\n\n\t"
        data = self.to_yaml().replace("\n", "\n\t")
        info += data
        return info.expandtabs(tabsize=4)

    @classmethod
    def read(cls, path):
        """Read the configuration from a YAML file."""
        config = read_yaml(path)
        return cls(**config)

    @classmethod
    def from_yaml(cls, config_str):
        """Create a configuration from a YAML string."""
        settings = yaml.safe_load(config_str)
        return cls(**settings)

    def write(self, path, overwrite=False):
        """Write the configuration to a YAML file."""
        path = make_path(path)

        if path.exists() and not overwrite:
            raise OSError(f"File exists already: {path}")

        path.write_text(self.to_yaml())

    def to_yaml(self):
        """Convert the configuration to a YAML string."""
        data = json.loads(self.model_dump_json())
        return yaml.dump(
            data, sort_keys=False, indent=4, width=80, default_flow_style=None
        )

    def set_logging(self):
        """Configure logging using the settings in ``general.log``."""
        self.general.log.level = self.general.log.level.upper()
        logging.basicConfig(**self.general.log.model_dump())
        log.info("Setting logging config: %r", self.general.log.model_dump())

    def update(self, config=None):
        """Update the configuration with the provided settings.

        Parameters
        ----------
        config : str or `CTAOAnalysisConfig`
            Configuration settings. A string is interpreted as YAML.

        Returns
        -------
        config : `CTAOAnalysisConfig`
            Updated configuration.
        """
        if isinstance(config, str):
            other = CTAOAnalysisConfig.from_yaml(config)
        elif isinstance(config, CTAOAnalysisConfig):
            other = config
        else:
            raise TypeError(f"Invalid type: {config}")

        config_new = deep_update(
            self.model_dump(exclude_defaults=True),
            other.model_dump(exclude_defaults=True),
        )
        return CTAOAnalysisConfig(**config_new)

    @staticmethod
    def _get_doc_sections():
        """Return documentation sections defined in the docs YAML file."""
        doc = defaultdict(str)
        with open(DOCS_FILE) as file:
            for line in filter(lambda line: not line.startswith("---"), file):
                line = line.strip("\n")
                if line.startswith("# Section: "):
                    keyword = line.replace("# Section: ", "")
                doc[keyword] += line + "\n"
        return doc
