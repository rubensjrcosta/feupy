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

__all__ = [
    "CTAOAnalysisConfig",
    "ROIAnalysisConfig",
]

CONFIG_PATH = Path(__file__).resolve().parent / "config"
DOCS_FILE = CONFIG_PATH / "docs.yaml"

log = logging.getLogger(__name__)


class ROIAnalysisConfig(GammapyBaseConfig):
    """Configuration for region-of-interest analyses."""

    general: GeneralConfig = GeneralConfig()
    roi: SpatialCircleConfig = SpatialCircleConfig()
    energy_range: EnergyRangeConfig = EnergyRangeConfig()

    def __str__(self):
        """Return the configuration as a readable YAML representation."""
        info = self.__class__.__name__ + "\n\n\t"
        data = self.to_yaml().replace("\n", "\n\t")
        info += data

        return info.expandtabs(tabsize=4)

    @classmethod
    def read(cls, path):
        """Read the configuration from a YAML file.

        Parameters
        ----------
        path : str or `~pathlib.Path`
            Configuration file path.

        Returns
        -------
        `ROIAnalysisConfig`
            Parsed configuration.
        """
        config = read_yaml(path)

        return cls(**config)

    @classmethod
    def from_yaml(cls, config_str):
        """Create a configuration from a YAML string.

        Parameters
        ----------
        config_str : str
            YAML configuration string.

        Returns
        -------
        `ROIAnalysisConfig`
            Parsed configuration.
        """
        settings = yaml.safe_load(config_str)

        return cls(**settings)

    def write(self, path, overwrite=False):
        """Write the configuration to a YAML file.

        Parameters
        ----------
        path : str or `~pathlib.Path`
            Destination file path.
        overwrite : bool, optional
            Whether to overwrite an existing file. Default is False.
        """
        path = make_path(path)

        if path.exists() and not overwrite:
            raise OSError(f"File exists already: {path}")

        path.write_text(self.to_yaml())

    def to_yaml(self):
        """Convert the configuration to a YAML string.

        Returns
        -------
        str
            YAML representation of the configuration.
        """
        data = json.loads(self.model_dump_json())

        return yaml.dump(
            data,
            sort_keys=False,
            indent=4,
            width=80,
            default_flow_style=None,
        )

    def set_logging(self):
        """Configure logging using the settings in ``general.log``."""
        self.general.log.level = self.general.log.level.upper()
        log_config = self.general.log.model_dump()

        logging.basicConfig(**log_config)
        log.info("Setting logging config: %r", log_config)

    def update(self, config=None):
        """Update the configuration with the provided settings.

        Parameters
        ----------
        config : str or `ROIAnalysisConfig`
            Configuration settings. A string is interpreted as YAML.

        Returns
        -------
        `ROIAnalysisConfig`
            Updated configuration.

        Raises
        ------
        TypeError
            If ``config`` has an unsupported type.
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

        with DOCS_FILE.open() as file:
            for line in filter(lambda line: not line.startswith("---"), file):
                line = line.rstrip("\n")

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
        """Return the configuration as a readable YAML representation."""
        info = self.__class__.__name__ + "\n\n\t"
        data = self.to_yaml().replace("\n", "\n\t")
        info += data

        return info.expandtabs(tabsize=4)

    @classmethod
    def read(cls, path):
        """Read the configuration from a YAML file.

        Parameters
        ----------
        path : str or `~pathlib.Path`
            Configuration file path.

        Returns
        -------
        `CTAOAnalysisConfig`
            Parsed configuration.
        """
        config = read_yaml(path)

        return cls(**config)

    @classmethod
    def from_yaml(cls, config_str):
        """Create a configuration from a YAML string.

        Parameters
        ----------
        config_str : str
            YAML configuration string.

        Returns
        -------
        `CTAOAnalysisConfig`
            Parsed configuration.
        """
        settings = yaml.safe_load(config_str)

        return cls(**settings)

    def write(self, path, overwrite=False):
        """Write the configuration to a YAML file.

        Parameters
        ----------
        path : str or `~pathlib.Path`
            Destination file path.
        overwrite : bool, optional
            Whether to overwrite an existing file. Default is False.
        """
        path = make_path(path)

        if path.exists() and not overwrite:
            raise OSError(f"File exists already: {path}")

        path.write_text(self.to_yaml())

    def to_yaml(self):
        """Convert the configuration to a YAML string.

        Returns
        -------
        str
            YAML representation of the configuration.
        """
        data = json.loads(self.model_dump_json())

        return yaml.dump(
            data,
            sort_keys=False,
            indent=4,
            width=80,
            default_flow_style=None,
        )

    def set_logging(self):
        """Configure logging using the settings in ``general.log``."""
        self.general.log.level = self.general.log.level.upper()
        log_config = self.general.log.model_dump()

        logging.basicConfig(**log_config)
        log.info("Setting logging config: %r", log_config)

    def update(self, config=None):
        """Update the configuration with the provided settings.

        Parameters
        ----------
        config : str or `CTAOAnalysisConfig`
            Configuration settings. A string is interpreted as YAML.

        Returns
        -------
        `CTAOAnalysisConfig`
            Updated configuration.

        Raises
        ------
        TypeError
            If ``config`` has an unsupported type.
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

        with DOCS_FILE.open() as file:
            for line in filter(lambda line: not line.startswith("---"), file):
                line = line.rstrip("\n")

                if line.startswith("# Section: "):
                    keyword = line.replace("# Section: ", "")

                doc[keyword] += line + "\n"

        return doc
