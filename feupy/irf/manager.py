# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""CTAO IRFs class."""

from __future__ import annotations

import os
import logging
from pathlib import Path
from functools import lru_cache
from typing import Dict, Tuple, Any

from gammapy.irf import load_irf_dict_from_file
from gammapy.data import observatory_locations

log = logging.getLogger(__name__)

IRFOption = Tuple[str, str, str, str]  # (array, azimuth, zenith, livetime)

__all__ = ["CTAOIRFManager"]

log = logging.getLogger(__name__)


class CTAOIRFManager:
    """
    Manager for CTAO IRFs (Prod5).
    """

    IRF_VERSION = "prod5 v0.1"

    _SITE_ARRAY = {
        "South": "14MSTs37SSTs",
        "South-SSTSubArray": "37SSTs",
        "South-MSTSubArray": "14MSTs",
        "North": "4LSTs09MSTs",
        "North-MSTSubArray": "09MSTs",
        "North-LSTSubArray": "4LSTs",
    }

    _OBS_TIME = {
        "0.5h": "1800s",
        "5h": "18000s",
        "50h": "180000s",
    }

    _BASE_PATH = Path(os.getenv("FEUPY_DATA", ".")) / "irfs/cta-prod5-zenodo-v0.1/fits"

    def __init__(self):
        self._cache: Dict[IRFOption, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    def get_irf(self, opt: IRFOption) -> Dict[str, Any]:
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