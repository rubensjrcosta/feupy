# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""ROI configuration."""
import logging
import json
import yaml

from pathlib import Path
from gammapy.utils.scripts import make_path, read_yaml
from pydantic.v1.utils import deep_update
from collections import defaultdict


from feupy.utils.config import (
    FeupyBaseConfig, 
    GeneralConfig, 
    EnergyRangeConfig,
    ROIConfig,
)


__all__ = ["AnalysisRoiConfig"]



CONFIG_PATH = Path(__file__).resolve().parent / "config"
DOCS_FILE = CONFIG_PATH / "docs.yaml"

log = logging.getLogger(__name__)


class AnalysisRoiConfig(FeupyBaseConfig):
    """Gammapy analysis configuration."""

    general: GeneralConfig = GeneralConfig()
    roi: ROIConfig = ROIConfig()
    energy_range: EnergyRangeConfig = EnergyRangeConfig()
#     plot: PlotConfig = PlotConfig()

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
        return AnalysisRoiConfig(**config)


    @classmethod
    def from_yaml(cls, config_str):
        """Create from YAML string."""
        settings = yaml.safe_load(config_str)
        return AnalysisRoiConfig(**settings)


    def write(self, path = None, overwrite=False):
        """Write to YAML file."""
        if path is not None:
            path = make_path(path)
        else: path = make_path(f"{self.general.config_file}")
            
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
            other = AnalysisRoiConfig.from_yaml(config)
        elif isinstance(config, AnalysisRoiConfig):
            other = config
        else:
            raise TypeError(f"Invalid type: {config}")

        config_new = deep_update(
            self.dict(exclude_defaults=True), other.dict(exclude_defaults=True)
        )
        return AnalysisRoiConfig(**config_new)


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