# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""VERITAS catalog and source classes."""

import logging
import os
import string

import numpy as np
from astropy.table import Table
from gammapy.catalog.core import SourceCatalog, SourceCatalogObject
from gammapy.datasets import Datasets, FluxPointsDataset
from gammapy.estimators import FluxPoints
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    LogParabolaSpectralModel,
    Models,
    PowerLawSpectralModel,
    SkyModel,
)
from gammapy.utils.scripts import make_path
from pandas import json_normalize

from feupy.utils.datasets import get_feupy_data_path
from feupy.utils.formatting import string_to_filename
from feupy.utils.io import read_yaml

log = logging.getLogger(__name__)

__all__ = [
    "SourceCatalogVTSCat",
    "SourceCatalogObjectVTSCat",
    "SourceCatalogVERITASCygnus",
    "SourceCatalogObjectVERITASCygnus",
]


def generate_unique_name(name, reference_id, unique_names):
    """Return a unique model name.

    Parameters
    ----------
    name : str
        Source name.
    reference_id : str
        Reference identifier.
    unique_names : collection of str
        Names already in use.

    Returns
    -------
    name : str
        Unique model name.
    """
    new_name = f"{name} ({reference_id[:7]})"
    if new_name not in unique_names:
        return new_name

    for letter in string.ascii_letters:
        new_name = f"{name} ({reference_id[:7]}-{letter})"
        if new_name not in unique_names:
            return new_name

    raise RuntimeError("Could not generate a unique model name.")


class SourceCatalogObjectVTSCat(SourceCatalogObject):
    """One source from the VTSCat catalog."""

    _DATASETS_PATH = get_feupy_data_path() / "catalogs/vtscat/datasets"
    _source_name_key = "source_name"

    def __str__(self):
        return self.info()

    def info(self, info="all"):
        """Return summary information for the source.

        Parameters
        ----------
        info : {"all", "basic", "position", "spectrum"}, optional
            Comma-separated information sections.

        Returns
        -------
        info : str
            Formatted source information.
        """
        if info == "all":
            info = "basic,position,spectrum"

        text = ""
        options = info.split(",")

        if "basic" in options:
            text += self._info_basic()
        if "position" in options:
            text += self._info_position()

        return text

    def _info_basic(self):
        """Return basic source information."""
        data = self.data
        return (
            "\n*** Basic info ***\n\n"
            f"Catalog row index (zero-based): {self.row_index}\n"
            f"Source name: {self.name}\n"
            f"VTSCat name: {data.veritas_name}\n"
            f"Common name: {data.common_name}\n"
            f"Other names: {data.other_names}\n"
            f"VTSCat id: {data.veritas_id}\n"
            f"Location: {data.where}\n"
            f"Type: {data.type}\n"
            f"VTSCat components: {data.veritas_components}\n"
            f"VTSCat ID name: {self.veritas_id}\n"
            f"Simbad ID: {data.simbad_id}\n"
            f"References: {data.reference_id}\n\n"
        )

    def _info_position(self):
        """Return source position information."""
        return (
            "\n*** Position info ***\n\n"
            f"RA: {self.data.ra:.3f}\n"
            f"DEC: {self.data.dec:.3f}\n"
        )

    def _spectral_model(self, table):
        """Fit and return a spectral model for a flux-points table."""
        reference = "1 TeV"
        spec_type = "pl"

        if spec_type == "lp":
            spec_model = LogParabolaSpectralModel(reference=reference)
        elif spec_type == "pl":
            spec_model = PowerLawSpectralModel(reference=reference)
        else:
            log.warning("Unknown spectral model type: %s", spec_type)
            return None

        flux_points = FluxPoints.from_table(table)
        dataset = FluxPointsDataset(data=flux_points)
        datasets = Datasets([dataset])

        model = SkyModel(spectral_model=spec_model)
        datasets.models = model

        Fit().run(datasets=datasets)
        return model.spectral_model

    def _sky_model(self, table):
        """Return a sky model for a flux-points table."""
        reference_id = table.meta.get("reference_id")
        model_name = generate_unique_name(self.name, reference_id, [])

        return SkyModel(
            spectral_model=self._spectral_model(table),
            name=model_name,
        )

    def flux_points(self):
        """Return source flux points as a list."""
        return [self._flux_points(table) for table in self.flux_points_tables]

    def _flux_points(self, table):
        """Return flux points for one table."""
        return FluxPoints.from_table(
            table=table,
            reference_model=self._sky_model(table),
            sed_type=table.meta.get("SED_TYPE"),
        )

    def _reference_id(self):
        """Return cleaned reference identifiers."""
        identifiers = self.data["reference_id"].split(", ")
        return [identifier.replace(" ", "") for identifier in identifiers]

    @property
    def veritas_id(self):
        """Return the formatted VERITAS identifier."""
        return f"VER-{self.data.veritas_id:06}"

    def _get_file_paths(self, reference_id, which="info"):
        """Return dataset files associated with a reference."""
        paths = []
        directory = make_path(f"{self._DATASETS_PATH}/{reference_id}")

        for filename in os.listdir(directory):
            path = make_path(f"{directory}/{filename}")

            if which == "info" and filename == "info.yaml":
                paths.append(path)
            elif which in {"sed", "lc"}:
                if (
                    filename.endswith(".ecsv")
                    and self.veritas_id in filename
                    and which in filename
                ):
                    paths.append(path)
            elif (
                which == "obs"
                and filename.endswith(".yaml")
                and self.veritas_id in filename
            ):
                paths.append(path)

        return paths

    @property
    def reference_id_info(self):
        """Return normalized metadata for source references."""
        data = []
        for reference_id in self._reference_id():
            for path in self._get_file_paths(reference_id, which="info"):
                data.append(read_yaml(path))
        return json_normalize(data)

    def get_observation_tables(self, reference_id):
        """Return observation metadata for a reference."""
        return [
            read_yaml(path) for path in self._get_file_paths(reference_id, which="obs")
        ]

    @property
    def observation_info(self):
        """Return normalized observation metadata."""
        data = []
        for reference_id in self._reference_id():
            data.extend(self.get_observation_tables(reference_id))
        return json_normalize(data)

    def get_flux_points_tables(self, reference_id):
        """Return flux-points tables for a reference."""
        tables = []

        for path in self._get_file_paths(reference_id, which="sed"):
            table = Table.read(make_path(path), format="ascii.ecsv")

            if "dnde" in table.colnames:
                table.meta["SED_TYPE"] = "dnde"
                if "dnde_ul" in table.colnames:
                    table["is_ul"] = [not np.isnan(value) for value in table["dnde_ul"]]

            if "e2dnde" in table.colnames:
                table.meta["SED_TYPE"] = "e2dnde"
                if "e2dnde_ul" in table.colnames:
                    table["is_ul"] = [
                        not np.isnan(value) for value in table["e2dnde_ul"]
                    ]

            tables.append(table)

        return tables

    @property
    def flux_points_tables(self):
        """Return all source flux-points tables."""
        tables = []
        for reference_id in self._reference_id():
            tables.extend(self.get_flux_points_tables(reference_id))
        return tables


class SourceCatalogVTSCat(SourceCatalog):
    """VTSCat catalog."""

    tag = "vtscat"
    bibcode = "2023RNAAS...7....6A"
    description = "VTSCat catalog from the VTSCat observatory"
    source_object_class = SourceCatalogObjectVTSCat

    def __init__(self, filename=None):
        """Initialize the VTSCat catalog.

        Parameters
        ----------
        filename : str or `~pathlib.Path`, optional
            Path to the VTSCat ECSV file.
        """
        if filename is None:
            filename = get_feupy_data_path() / "catalogs/vtscat/sources/vtscat.ecsv"

        table = Table.read(make_path(filename), format="ascii.ecsv")
        super().__init__(
            table=table,
            source_name_key="source_name",
        )


class SourceCatalogObjectVERITASCygnus(SourceCatalogObject):
    """One source from the VERITAS Cygnus catalog."""

    _source_name_key = "source_name"
    _DATA_PATH = "$FEUPY_DATA/catalogs/veritas/"
    _MODELS = None

    def __str__(self):
        return self.info()

    def info(self, info="all"):
        """Return summary information for the source.

        Parameters
        ----------
        info : {"all", "basic", "position", "spectrum"}, optional
            Comma-separated information sections.

        Returns
        -------
        info : str
            Formatted source information.
        """
        if info == "all":
            info = "basic,position,spectrum"

        text = ""
        options = info.split(",")

        if "basic" in options:
            text += self._info_basic()
        if "position" in options:
            text += self._info_position()
        if "spectrum" in options:
            text += self._info_spectrum()

        return text

    def _info_basic(self):
        """Return basic source information."""
        return (
            "\n*** Basic info ***\n\n"
            f"Catalog row index (zero-based): {self.row_index}\n"
            f"Source name: {self.name}\n"
        )

    def _info_position(self):
        """Return source position information."""
        return (
            "\n*** Position info ***\n\n"
            f"RA: {self.data.ra:.3f}\n"
            f"DEC: {self.data.dec:.3f}\n"
        )

    def _info_spectrum(self):
        """Return spectral information."""
        model = self.spectral_model()
        if model is None:
            return "\n*** Spectral info ***\n\nNo spectrum available"

        text = "\n*** Spectral info ***\n\n"
        text += f"Spectrum type: {model.tag[0]}\n"

        for parameter in model.parameters:
            try:
                unit = f"{parameter.unit:unicode}"
            except AttributeError:
                unit = ""

            text += (
                f"{parameter.name}: {parameter.value:.3f} ± "
                f"{parameter.error:.3f} {unit}\n"
            )

        return text

    @classmethod
    def _get_models(cls):
        """Load and cache VERITAS source models."""
        if cls._MODELS is None:
            filename = make_path(f"{cls._DATA_PATH}/models.yaml")
            cls._MODELS = Models.read(filename)
        return cls._MODELS

    def spectral_model(self):
        """Return the source spectral model."""
        return self._get_models()[self.name].spectral_model

    def spatial_model(self):
        """Return the source spatial model."""
        return self._get_models()[self.name].spatial_model

    def sky_model(self):
        """Return the source sky model."""
        return self._get_models()[self.name]

    @property
    def flux_points(self):
        """Return source flux points."""
        filename = make_path(f"{self._DATA_PATH}/{string_to_filename(self.name)}.fits")
        return FluxPoints.read(filename)


class SourceCatalogVERITASCygnus(SourceCatalog):
    """VERITAS Cygnus catalog."""

    tag = "veritas-2018ApJ"
    bibcode = "2018ApJ...861..134A"
    description = (
        "A Very High Energy gamma-ray survey toward the Cygnus region of the Galaxy"
    )
    source_object_class = SourceCatalogObjectVERITASCygnus

    def __init__(self, filename=None):
        """Initialize the VERITAS Cygnus catalog.

        Parameters
        ----------
        filename : str or `~pathlib.Path`, optional
            Path to the VERITAS FITS catalog.
        """
        if filename is None:
            filename = get_feupy_data_path() / "catalogs/veritas/veritas.fits"

        table = Table.read(make_path(filename))
        super().__init__(
            table=table,
            source_name_key="source_name",
        )
