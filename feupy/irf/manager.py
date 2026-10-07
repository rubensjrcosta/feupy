# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""CTAO IRFs class."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any

from gammapy.data import observatory_locations
from gammapy.irf import load_irf_dict_from_file

log = logging.getLogger(__name__)

IRFOption = tuple[str, str, str, str]  # (array, azimuth, zenith, livetime)

__all__ = ["CTAOIRFManager"]

log = logging.getLogger(__name__)


class CTAOIRFManager:
    """
    Manager for CTAO IRFs (Prod5).
    """

    IRF_VERSION = "prod5 v0.1"

    _PROD5_SITE_ARRAY = {
        "South": "14MSTs37SSTs",
        "South-SSTSubArray": "37SSTs",
        "South-MSTSubArray": "14MSTs",
        "North": "4LSTs09MSTs",
        "North-MSTSubArray": "09MSTs",
        "North-LSTSubArray": "4LSTs",
    }

    _PROD6_SITE_ARRAY = {
        "South": "2LSTs14MSTs37SSTs",
        "North": "4LSTs09MSTs",
    }
    
    _OBS_TIME = {
        "0.5h": "1800s",
        "5h": "18000s",
        "50h": "180000s",
    }

    _AZIMUTHS = [
        "AverageAz",
        "NorthAz",
        "SouthAz",
    ]

    _PROD5_ZENITHS = [
        "20deg",
        "40deg",
        "60deg",
    ]
    
    _PROD6_ZENITHS = [
        "20deg",
        "40deg",
        "52deg",
        "60deg",
    ]

    _DATA_PATH = Path(
            os.getenv("FEUPY_DATA", ".")
        )
    
    _PROD5_BASE_PATH = (
        _DATA_PATH
        / "irfs"
        / "cta-prod5-zenodo-v0.1"
        / "fits"
    )
    
    _PROD6_BASE_PATH = (
        _DATA_PATH
        / "irfs"
        / "ctao-prod6-zenodo-v1.0"
        / "fits"
    )

    def __init__(
        self,
        production: str = "prod5",
        condition: str = "dark",
    ):
        production = production.lower()
        condition = condition.lower()
    
        if production not in {
            "prod5",
            "prod6",
        }:
            raise ValueError(
                "Invalid CTAO IRF production "
                f"{production!r}. "
                "Available productions are: "
                "'prod5', 'prod6'."
            )
    
        if (
            production == "prod6"
            and condition not in {
                "dark",
                "halfmoon",
            }
        ):
            raise ValueError(
                "Invalid Prod6 observing condition "
                f"{condition!r}. "
                "Available conditions are: "
                "'dark', 'halfmoon'."
            )
    
        self.production = production
        self.condition = condition
    
        self._cache: dict[
            IRFOption,
            dict[str, Any],
        ] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def site_arrays(self):
        """Return array configurations for the selected production."""
        if self.production == "prod5":
            return self._PROD5_SITE_ARRAY
    
        return self._PROD6_SITE_ARRAY
    
    
    @property
    def zeniths(self):
        """Return zenith angles for the selected production."""
        if self.production == "prod5":
            return self._PROD5_ZENITHS
    
        return self._PROD6_ZENITHS
    
    
    @property
    def base_path(self):
        """Return base path for the selected production."""
        if self.production == "prod5":
            return self._PROD5_BASE_PATH
    
        return self._PROD6_BASE_PATH

    @staticmethod
    def _array_label(name: str) -> str:
        return name.replace("SubArray", "s")

    @staticmethod
    def _get_observatory(opt: IRFOption):
        return (
            observatory_locations["cta_south"]
            if "South" in opt[0]
            else observatory_locations["cta_north"]
        )

    @classmethod
    def _build_path(cls, opt: IRFOption) -> Path:
        array, az, zen, lt = opt
        site = array.split("-")[0]

        subdir = f"CTA-Performance-prod5-v0.1-{array}-{zen}.FITS"

        filename = (
            f"Prod5-{site}-{zen}-{az}-"
            f"{cls._SITE_ARRAY[array]}."
            f"{cls._OBS_TIME[lt]}-v0.1.fits.gz"
        )

        return cls._BASE_PATH / subdir / filename

    # ------------------------------------------------------------------
    # File loader (cached)
    # ------------------------------------------------------------------

    @staticmethod
    @lru_cache(maxsize=128)
    def _load_file(path: Path):
        log.debug(f"Loading IRF: {path}")
        return load_irf_dict_from_file(str(path))

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def get_irf(self, opt: IRFOption) -> dict[str, Any]:
        """
        Load IRF and return metadata dict.
        """

        # -----------------------------
        # Safety check (IMPORTANT FIX)
        # -----------------------------
        if isinstance(opt, list):
            opt = tuple(opt)

        if not isinstance(opt, tuple):
            raise TypeError(f"IRFOption must be tuple, got {type(opt)}")

        # -----------------------------
        # Cache
        # -----------------------------
        if opt in self._cache:
            return self._cache[opt]

        path = self._build_path(opt)
        irf = self._load_file(path)

        meta = {
            "irf": irf,
            "label": self._make_label(opt, which="both"),  # FIXED BUG
            "name": self._make_name(opt),
            "file_path": path,
            "obs_location": self._get_observatory(opt),
            "option": opt,
        }

        self._cache[opt] = meta
        return meta

    # ------------------------------------------------------------------
    # Naming
    # ------------------------------------------------------------------

    @staticmethod
    def _make_label(opt: IRFOption, which: str = "both") -> str:
        array, az, zen, lt = opt

        ss = "CTAO "
        array_label = CTAOIRFManager._array_label(array)
        azimuth_label = az.replace("AverageAz", "")

        if which == "zenith":
            extra = f" ({zen})"
        elif which == "livetime":
            extra = f" ({lt})"
        elif which == "both":
            extra = f" ({zen}-{lt})"
        else:
            extra = ""

        return f"{ss}{array_label}{azimuth_label}{extra}"

    @staticmethod
    def _make_name(opt: IRFOption) -> str:
        array, az, zen, lt = opt
        return f"CTAO-{array}_{zen}_{lt}"

    @classmethod
    def get_irfs_options(cls):
        """Return all available IRF combinations."""
        return [
            (array, azimuth, zenith, livetime)
            for array, azimuth, zenith, livetime in product(
                cls._SITE_ARRAY,
                cls._AZIMUTHS,
                cls._ZENITHS,
                cls._OBS_TIME,
            )
        ]
