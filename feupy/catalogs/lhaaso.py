# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""LHAASO catalog utilities and source classes."""

import logging

import astropy.units as u
import numpy as np
from astropy.table import Table
from gammapy.catalog.core import SourceCatalog, SourceCatalogObject
from gammapy.estimators import FluxPoints
from gammapy.modeling.models import (
    LogParabolaSpectralModel,
    Models,
    PowerLawSpectralModel,
    SkyModel,
)
from gammapy.utils.scripts import make_path

from feupy.utils.fitting import fit_spectral_model_to_flux_points
from feupy.utils.formatting import string_to_filename
from feupy.utils.tables.utils import remove_nan_rows

log = logging.getLogger(__name__)

__all__ = [
    "create_flux_points_table_1lhaaso",
    "get_flux_points_1lhaaso",
    "SourceCatalogLHAASO",
    "SourceCatalogObjectLHAASO",
    "SourceCatalogObjectExtraLHAASO",
    "SourceCatalogExtraLHAASO",
]


def create_flux_points_table_1lhaaso(source, which):
    """Create a flux-points table for a 1LHAASO catalog source.

    Parameters
    ----------
    source : `~gammapy.catalog.SourceCatalogObject1LHAASO`
        Source from the 1LHAASO catalog.
    which : {"point", "extended"}
        Source model component to use.

    Returns
    -------
    table : `~astropy.table.Table`
        Flux-points table for the selected model component.

    Raises
    ------
    ValueError
        If the requested model component is not available.
    """

    def get_model_tag(source, which):
        if which in source.data["Model_a"]:
            return ""
        if which in source.data["Model_b"]:
            return "_b"
        raise ValueError("Invalid model component name")

    def parse_value(source, name, which):
        tag = get_model_tag(source, which)
        value = u.Quantity(source.data[f"{name}{tag}"])
        is_ul = False

        if (
            np.isnan(value) or value == 0 * value.unit
        ) and f"{name}_ul{tag}" in source.data:
            value = source.data[f"{name}_ul{tag}"]
            is_ul = True

        return value, is_ul

    def get_value(source, name, which):
        value, _ = parse_value(source, name, which)
        return value

    e_ref = u.Quantity([get_value(source, "E0", which)])
    spec_model = source.spectral_model(which=which)
    dnde = spec_model(e_ref)
    dnde_err = spec_model.evaluate_error(e_ref)[1]

    is_ul = False
    dnde_ul = np.nan
    if not dnde_err.value:
        is_ul = True
        dnde_ul = dnde

    table = Table()
    table["e_ref"] = e_ref
    table["e_ref"].description = "Reference energy"

    table["dnde"] = dnde
    table["dnde"].description = "Differential flux at reference energy"

    table["dnde_err"] = dnde_err
    table["dnde_err"].description = "Error on the differential flux"

    table["dnde_ul"] = dnde_ul
    table["dnde_ul"].unit = dnde.unit
    table["dnde_ul"].description = "Upper limit for differential flux"

    table["is_ul"] = is_ul
    table["is_ul"].description = "Whether the data point is an upper limit"

    table.meta["source_name"] = source.name
    table.meta["SED_TYPE"] = "dnde"
    table.meta["model"] = which
    table.meta["comments"] = [
        "Reference: https://iopscience.iop.org/article/10.3847/1538-4365/acfd29"
    ]

    for column in table.colnames:
        if column.startswith("dnde"):
            table[column].format = ".3e"
        elif column.startswith("e_"):
            table[column].format = ".3f"

    return table


def get_flux_points_1lhaaso(source, which):
    """Create flux points for a 1LHAASO catalog source.

    Parameters
    ----------
    source : `~gammapy.catalog.SourceCatalogObject1LHAASO`
        Source from the 1LHAASO catalog.
    which : {"point", "extended"}
        Source model component to use.

    Returns
    -------
    flux_points : `~gammapy.estimators.FluxPoints`
        Flux points for the selected model component.
    """
    table = create_flux_points_table_1lhaaso(source, which)
    return FluxPoints.from_table(
        table=table,
        reference_model=source.spectral_model(which),
        sed_type=table.meta["SED_TYPE"],
    )


class SourceCatalogObjectExtraLHAASO(SourceCatalogObject):
    """One source from the dedicated LHAASO catalog."""

    _MODELS = None
    _source_name_key = "source_name"

    def __str__(self):
        return self.info()

    def info(self, info="all"):
        """Return summary information for the source.

        Parameters
        ----------
        info : {"all", "basic", "position", "spectrum"}, optional
            Information sections to include.

        Returns
        -------
        info : str
            Formatted source information.
        """
        details = {
            "basic": self._info_basic,
            "position": self._info_position,
            "spectrum": self._info_spectrum,
        }
        selected = info.split(",") if info != "all" else details.keys()
        return "\n".join(details[item]() for item in selected if item in details)

    def _info_basic(self):
        """Return basic source information."""
        return (
            "\n*** Basic info ***\n\n"
            f"Catalog row index: {self.row_index}\n"
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
            return "No spectral information available."

        lines = [
            "\n*** Spectral info ***\n",
            f"Spectrum type: {model.tag[0]}",
        ]
        lines.extend(
            f"{par.name}: {par.value:.3f} ± {par.error} {par.unit if par.unit else ''}"
            for par in model.parameters
        )
        return "\n".join(lines)

    def spectral_model(self):
        """Return the spectral model associated with the source."""
        if self._MODELS is None:
            filename = (
                "$FEUPY_DATA/dedicated_publications/lhaaso/"
                "2024icrc.confE.643Y/models.yaml"
            )
            self.__class__._MODELS = Models.read(make_path(filename))

        if self.name in self._MODELS.names:
            return self._MODELS[self.name].spectral_model

        return None

    def sky_model(self):
        """Return the source sky model."""
        spectral_model = self.spectral_model()
        if spectral_model is None:
            return None
        return SkyModel(spectral_model=spectral_model, name=self.name)

    @property
    def flux_points(self):
        """Return source flux points."""
        filename = (
            "$FEUPY_DATA/dedicated_publications/lhaaso/2024icrc.confE.643Y/"
            f"{string_to_filename(self.name)}.fits"
        )
        filename = make_path(filename)

        if not filename.exists():
            return None

        return FluxPoints.read(
            filename,
            reference_model=self.sky_model(),
            sed_type="e2dnde",
        )


class SourceCatalogExtraLHAASO(SourceCatalog):
    """Catalog for the dedicated LHAASO publication."""

    tag = "LHAASO-2024icrc"
    bibcode = "2024icrc.confE.643Y"
    description = "LHAASO first 12 PeVatrons Catalogue"
    source_object_class = SourceCatalogObjectExtraLHAASO

    def __init__(
        self,
        filename=(
            "$FEUPY_DATA/dedicated_publications/lhaaso/2024icrc.confE.643Y/lhaaso_2024_catalog.ecsv"
        ),
    ):
        """Initialize the dedicated LHAASO catalog.

        Parameters
        ----------
        filename : str or `~pathlib.Path`, optional
            Path to the catalog ECSV file.
        """
        table = Table.read(make_path(filename), format="ascii.ecsv")
        super().__init__(table=table, source_name_key="source_name")


class SourceCatalogObjectLHAASO(SourceCatalogObject):
    """One source from the LHAASO first 12 PeVatrons catalog."""

    _source_name_key = "source_name"
    _sed_type = "e2dnde"

    def __str__(self):
        return self.info()

    def info(self, info="all"):
        """Return summary information for the source."""
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
            unit = f"{parameter.unit:unicode}" if parameter.unit else ""
            text += (
                f"{parameter.name}: {parameter.value:.3f} ± {parameter.error} {unit}\n"
            )

        return text

    def spectral_model(self):
        """Fit and return the source spectral model."""
        spec_type = self.data["spec_type"]
        reference = self.data["spec_reference"]

        if spec_type == "lp":
            model = LogParabolaSpectralModel(reference=reference)
        elif spec_type == "pl":
            model = PowerLawSpectralModel(reference=reference)
        else:
            log.warning("Unknown spectral model type: %s", spec_type)
            return None

        return fit_spectral_model_to_flux_points(
            self.flux_points_table,
            model,
        )

    def sky_model(self):
        """Return the source sky model."""
        spectral_model = self.spectral_model()

        if spectral_model is None:
            return None

        return SkyModel(
            spectral_model=spectral_model,
            name=self.name,
        )

    @property
    def flux_points(self):
        """Return source flux points."""
        return FluxPoints.from_table(
            table=self.flux_points_table,
            reference_model=self.sky_model(),
            sed_type=self._sed_type,
        )

    @property
    def flux_points_table(self):
        """Return the source flux-points table."""
        table = Table()
        table.meta["SED_TYPE"] = self._sed_type

        for key in self.data:
            if not key.startswith("sed_"):
                continue

            values = self.data[key]
            array = np.asarray(getattr(values, "value", values))

            if array.dtype.kind in "fc" and np.all(np.isnan(array)):
                continue

            table[key.removeprefix("sed_")] = values

        return remove_nan_rows(table)


class SourceCatalogLHAASO(SourceCatalog):
    """LHAASO first 12 PeVatrons catalog."""

    tag = "LHAASO"
    bibcode = "2021Natur.594...33C"
    description = "LHAASO first 12 PeVatrons Catalogue"
    source_object_class = SourceCatalogObjectLHAASO

    def __init__(
        self,
        filename="$FEUPY_DATA/catalogs/lhaaso/lhaaso_catalog.ecsv",
    ):
        """Initialize the LHAASO catalog.

        Parameters
        ----------
        filename : str or `~pathlib.Path`, optional
            Path to the LHAASO ECSV catalog.
        """
        table = Table.read(make_path(filename), format="ascii.ecsv")
        super().__init__(table=table, source_name_key="source_name")
