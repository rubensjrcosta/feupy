# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""CTAO instrument response functions."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any

from gammapy.data import observatory_locations
from gammapy.irf import load_irf_dict_from_file

__all__ = ["CTAOIRFManager"]

log = logging.getLogger(__name__)


IRFOption = tuple[str, str, str, str]
# (array, azimuth, zenith, livetime)


class CTAOIRFManager:
    """Manager for CTAO instrument response functions.

    Parameters
    ----------
    production : {"prod5", "prod6"}
        CTAO IRF production. Default is ``"prod5"``.
    condition : {"dark", "halfmoon"}
        Observing condition. This option is used by Prod6.
        Default is ``"dark"``.
    """

    _PROD5_VERSION = "v0.1"
    _PROD6_VERSION = "v1.0"

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

    _PROD5_OBS_TIME = {
        "0.5h": "1800s",
        "5h": "18000s",
        "50h": "180000s",
    }

    _PROD6_OBS_TIME = {
        "100s": "100s",
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
        os.getenv(
            "FEUPY_DATA",
            ".",
        )
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
    # Production configuration
    # ------------------------------------------------------------------

    @property
    def version(self) -> str:
        """Return version of the selected IRF production."""
        if self.production == "prod5":
            return self._PROD5_VERSION

        return self._PROD6_VERSION

    @property
    def site_arrays(self) -> dict[str, str]:
        """Return array configurations for the selected production."""
        if self.production == "prod5":
            return self._PROD5_SITE_ARRAY

        return self._PROD6_SITE_ARRAY

    @property
    def observation_times(self) -> dict[str, str]:
        """Return observation times for the selected production."""
        if self.production == "prod5":
            return self._PROD5_OBS_TIME

        return self._PROD6_OBS_TIME

    @property
    def zeniths(self) -> list[str]:
        """Return zenith angles for the selected production."""
        if self.production == "prod5":
            return self._PROD5_ZENITHS

        return self._PROD6_ZENITHS

    @property
    def base_path(self) -> Path:
        """Return base path for the selected production."""
        if self.production == "prod5":
            return self._PROD5_BASE_PATH

        return self._PROD6_BASE_PATH

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _array_label(name: str) -> str:
        """Return a human-readable array label."""
        return name.replace(
            "SubArray",
            "s",
        )

    @staticmethod
    def _get_observatory(opt: IRFOption):
        """Return observatory location for an IRF option."""
        return (
            observatory_locations["cta_south"]
            if "South" in opt[0]
            else observatory_locations["cta_north"]
        )

    # ------------------------------------------------------------------
    # Path builders
    # ------------------------------------------------------------------

    def _build_prod5_path(
        self,
        opt: IRFOption,
    ) -> Path:
        """Build the file path for a Prod5 IRF."""
        array, azimuth, zenith, livetime = opt

        site = array.split("-")[0]

        subdir = (
            f"CTA-Performance-prod5-v0.1-"
            f"{array}-{zenith}.FITS"
        )

        filename = (
            f"Prod5-{site}-{zenith}-{azimuth}-"
            f"{self._PROD5_SITE_ARRAY[array]}."
            f"{self._PROD5_OBS_TIME[livetime]}-"
            f"v0.1.fits.gz"
        )

        return (
            self._PROD5_BASE_PATH
            / subdir
            / filename
        )

    def _build_prod6_path(
        self,
        opt: IRFOption,
    ) -> Path:
        """Build the file path for a Prod6 IRF."""
        array, azimuth, zenith, livetime = opt

        subdir = (
            f"CTAO-Performance-Prod6-CTAO-"
            f"{array}-{zenith}-"
            f"{self.condition}-v1.0.FITS"
        )

        filename = (
            f"Prod6-CTAO-{array}-{zenith}-{azimuth}-"
            f"{self._PROD6_SITE_ARRAY[array]}-"
            f"{self.condition}-"
            f"{self._PROD6_OBS_TIME[livetime]}-"
            f"v1.0.fits.gz"
        )

        return (
            self._PROD6_BASE_PATH
            / subdir
            / filename
        )

    def _build_path(
        self,
        opt: IRFOption,
    ) -> Path:
        """Build the IRF file path for the selected production."""
        if self.production == "prod5":
            return self._build_prod5_path(opt)

        return self._build_prod6_path(opt)

    # ------------------------------------------------------------------
    # File loader
    # ------------------------------------------------------------------

    @staticmethod
    @lru_cache(maxsize=128)
    def _load_file(
        path: Path,
    ):
        """Load an IRF file with caching."""
        log.debug(
            "Loading IRF: %s",
            path,
        )

        return load_irf_dict_from_file(
            str(path)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_irf(
        self,
        opt: IRFOption,
    ) -> dict[str, Any]:
        """Load an IRF and return its metadata."""
        if isinstance(opt, list):
            opt = tuple(opt)

        if not isinstance(opt, tuple):
            raise TypeError(
                "IRFOption must be tuple, "
                f"got {type(opt)}"
            )

        if opt in self._cache:
            return self._cache[opt]

        path = self._build_path(opt)
        irf = self._load_file(path)

        meta = {
            "irf": irf,
            "label": self._make_label(
                opt,
                which="both",
            ),
            "name": self._make_name(opt),
            "file_path": path,
            "obs_location": self._get_observatory(opt),
            "option": opt,
            "production": self.production,
            "version": self.version,
        }

        if self.production == "prod6":
            meta["condition"] = self.condition

        self._cache[opt] = meta

        return meta

    # ------------------------------------------------------------------
    # Naming
    # ------------------------------------------------------------------

    @staticmethod
    def _make_label(
        opt: IRFOption,
        which: str = "both",
    ) -> str:
        """Create a human-readable IRF label."""
        array, azimuth, zenith, livetime = opt

        prefix = "CTAO "

        array_label = CTAOIRFManager._array_label(
            array
        )

        azimuth_label = azimuth.replace(
            "AverageAz",
            "",
        )

        if which == "zenith":
            extra = f" ({zenith})"

        elif which == "livetime":
            extra = f" ({livetime})"

        elif which == "both":
            extra = (
                f" ({zenith}-{livetime})"
            )

        else:
            extra = ""

        return (
            f"{prefix}"
            f"{array_label}"
            f"{azimuth_label}"
            f"{extra}"
        )

    @staticmethod
    def _make_name(
        opt: IRFOption,
    ) -> str:
        """Create an IRF name."""
        array, _, zenith, livetime = opt

        return (
            f"CTAO-{array}_"
            f"{zenith}_"
            f"{livetime}"
        )

    # ------------------------------------------------------------------
    # Available options
    # ------------------------------------------------------------------

    def get_irfs_options(
        self,
    ) -> list[IRFOption]:
        """Return all available IRF combinations."""
        return [
            (
                array,
                azimuth,
                zenith,
                livetime,
            )
            for (
                array,
                azimuth,
                zenith,
                livetime,
            ) in product(
                self.site_arrays,
                self._AZIMUTHS,
                self.zeniths,
                self.observation_times,
            )
        ]