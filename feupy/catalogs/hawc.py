# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""HAWC catalog utilities and source classes."""

import numpy as np
from astropy import units as u
from astropy.table import Column, Table
from gammapy.catalog.core import SourceCatalog, SourceCatalogObject
from gammapy.catalog.hawc import SourceCatalog2HWC, SourceCatalog3HWC
from gammapy.estimators import FluxPoints
from gammapy.modeling.models import Models, SkyModel
from gammapy.utils.scripts import make_path

from feupy.utils.formatting import string_to_filename

__all__ = [
    "create_flux_points_table_3hwc",
    "get_flux_points_3hwc",
    "create_flux_points_table_2hwc",
    "get_flux_points_2hwc",
    "SourceCatalogObjectEHWC",
    "SourceCatalogEHWC",
    "SourceCatalogObjectExtraHAWC",
    "SourceCatalogExtraHAWC",
]


def create_flux_points_table_3hwc(source):
    """Create a flux-points table for a 3HWC catalog source.

    Parameters
    ----------
    source : `~gammapy.catalog.SourceCatalogObject3HWC`
        Source from the 3HWC catalog.

    Returns
    -------
    table : `~astropy.table.Table`
        Flux-points table containing differential flux values, uncertainties,
        and upper-limit information.
    """
    catalog = SourceCatalog3HWC()
    data = source.data

    table = Table()
    table.meta["source_name"] = source.name
    table.meta["catalog_name"] = catalog.table.meta["catalog_name"]
    table.meta["SED_TYPE"] = "dnde"
    table.meta["search_radius"] = data["search_radius"]
    table.meta["spec0_radius"] = data["spec0_radius"]
    table.meta["reference"] = catalog.table.meta["reference"]

    e_ref = Column(
        name="e_ref",
        data=u.Quantity([7 * u.TeV]),
        description="Reference energy",
        format=".3g",
    )
    dnde = Column(
        name="dnde",
        data=u.Quantity([data["spec0_dnde"]]),
        description=catalog.table["spec0_dnde"].description,
        format=catalog.table["spec0_dnde"].format,
    )
    dnde_errn = Column(
        name="dnde_errn",
        data=u.Quantity([-data["spec0_dnde_errn"]]),
        description=catalog.table["spec0_dnde_errn"].description,
        format=catalog.table["spec0_dnde_errn"].format,
    )
    dnde_errp = Column(
        name="dnde_errp",
        data=u.Quantity([data["spec0_dnde_errp"]]),
        description=catalog.table["spec0_dnde_errp"].description,
        format=catalog.table["spec0_dnde_errp"].format,
    )
    dnde_ul = Column(
        name="dnde_ul",
        data=[np.nan],
        unit=data["spec0_dnde"].unit,
        description="Differential flux upper limit",
    )
    is_ul = Column(
        name="is_ul",
        data=[False],
        description="Whether the data point is an upper limit",
        dtype=bool,
    )

    table.add_columns([e_ref, dnde, dnde_errn, dnde_errp, dnde_ul, is_ul])
    return table


def get_flux_points_3hwc(source):
    """Create flux points for a 3HWC catalog source.

    Parameters
    ----------
    source : `~gammapy.catalog.SourceCatalogObject3HWC`
        Source from the 3HWC catalog.

    Returns
    -------
    flux_points : `~gammapy.estimators.FluxPoints`
        Flux points for the source.
    """
    table = create_flux_points_table_3hwc(source)
    return FluxPoints.from_table(
        table,
        sed_type=table.meta["SED_TYPE"],
        reference_model=source.spectral_model(),
    )


def create_flux_points_table_2hwc(source, which="point"):
    """Create a flux-points table for a 2HWC catalog source.

    Parameters
    ----------
    source : `~gammapy.catalog.SourceCatalogObject2HWC`
        Source from the 2HWC catalog.
    which : {"point", "extended"}, optional
        Source model to use.

    Returns
    -------
    table : `~astropy.table.Table`
        Flux-points table for the selected source model.

    Raises
    ------
    ValueError
        If ``which="extended"`` is requested but no extended model is available.
    """
    catalog = SourceCatalog2HWC()

    if which == "extended" and source.n_models != 2:
        raise ValueError("No extended model available for this source.")

    e_ref = u.Quantity([7 * u.TeV])
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
    table["dnde"].description = "Differential flux SED values"

    table["dnde_err"] = dnde_err
    table["dnde_err"].description = "Differential flux SED errors"

    table["dnde_ul"] = dnde_ul
    table["dnde_ul"].description = "Differential flux SED upper limit"

    table["is_ul"] = is_ul
    table["is_ul"].description = "Whether the data point is an upper limit"

    table.meta["source_name"] = source.name
    table.meta["catalog_name"] = catalog.table.meta["catalog_name"]
    table.meta["SED_TYPE"] = "dnde"
    table.meta["reference"] = catalog.table.meta["reference"]

    for column in table.colnames:
        if column.startswith("dnde"):
            table[column].format = ".3e"
        elif column.startswith("e_"):
            table[column].format = ".3f"

    return table


def get_flux_points_2hwc(source, which="point"):
    """Create flux points for a 2HWC catalog source.

    Parameters
    ----------
    source : `~gammapy.catalog.SourceCatalogObject2HWC`
        Source from the 2HWC catalog.
    which : {"point", "extended"}, optional
        Source model to use.

    Returns
    -------
    flux_points : `~gammapy.estimators.FluxPoints`
        Flux points for the selected source model.
    """
    table = create_flux_points_table_2hwc(source, which=which)
    return FluxPoints.from_table(
        table,
        sed_type=table.meta["SED_TYPE"],
        reference_model=source.spectral_model(which=which),
    )


class SourceCatalogObjectEHWC(SourceCatalogObject):
    """One source from the eHWC catalog."""

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
            filename = "$FEUPY_DATA/catalogs/ehwc/models.yaml"
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
        """Flux points as a `~gammapy.estimators.FluxPoints` object."""
        table = self.flux_points_table
        if table is None:
            return None
        return FluxPoints.from_table(table=table, sed_type="e2dnde")

    def _add_source_meta(self, table):
        """Add source metadata to a flux-points table."""
        catalog = SourceCatalogEHWC()
        table.meta["source_name"] = self.name
        table.meta["catalog_name"] = catalog.table.meta["catalog_name"]
        table.meta["SED_TYPE"] = "e2dnde"
        table.meta["comments"] = catalog.table.meta["comments"]

    @property
    def flux_points_table(self):
        """Return the source differential flux-points table."""
        data = self.data
        table = Table()
        self._add_source_meta(table)

        valid = np.isfinite(data["sed_e_ref"].value)
        if valid.sum() == 0:
            return None

        table["e_ref"] = data["sed_e_ref"]
        table["e_ref"].description = "Reference energy"

        table["e2dnde"] = data["sed_e2dnde"]
        table["e2dnde"].description = "Differential flux SED values"

        table["e2dnde_errn"] = data["sed_e2dnde_errn"]
        table["e2dnde_errn"].description = "Differential flux SED negative errors"

        table["e2dnde_errp"] = data["sed_e2dnde_errp"]
        table["e2dnde_errp"].description = "Differential flux SED positive errors"

        table["e2dnde_ul"] = data["sed_e2dnde_ul"]
        table["e2dnde_ul"].description = "Differential flux SED upper limit"

        table["is_ul"] = data["sed_is_ul"]
        table["is_ul"].description = "Upper-limit indicator"

        for column in table.colnames:
            if column.startswith("e2dnde"):
                table[column].format = ".3e"
            elif column.startswith("e_"):
                table[column].format = ".3f"

        table = table[valid]

        for column in list(table.colnames):
            if not np.isfinite(table[column]).any():
                table.remove_column(column)

        return table


class SourceCatalogEHWC(SourceCatalog):
    """HAWC eHWC catalog.

    References
    ----------
    https://doi.org/10.1103/PhysRevLett.124.021102
    """

    tag = "ehwc"
    bibcode = "2020PhRvL.124b1102A"
    description = "Extra HAWC catalog data"
    source_object_class = SourceCatalogObjectEHWC

    def __init__(
        self,
        filename="$FEUPY_DATA/catalogs/ehwc/ehwc_catalog.ecsv",
    ):
        """Initialize the eHWC catalog.

        Parameters
        ----------
        filename : str or `~pathlib.Path`, optional
            Path to the eHWC ECSV catalog.
        """
        table = Table.read(make_path(filename), format="ascii.ecsv")
        super().__init__(table=table, source_name_key="source_name")


class SourceCatalogObjectExtraHAWC(SourceCatalogObject):
    """One source from the dedicated HAWC J1825-134 publication catalog."""

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
                "$FEUPY_DATA/dedicated_publications/hawc/"
                "2021ApJ...907L..30A/models.yaml"
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
            "$FEUPY_DATA/dedicated_publications/hawc/2021ApJ...907L..30A/"
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


class SourceCatalogExtraHAWC(SourceCatalog):
    """Catalog for the dedicated HAWC J1825-134 publication.

    References
    ----------
    https://iopscience.iop.org/article/10.3847/2041-8213/abd77b
    """

    tag = "hwc-2021ApJ"
    bibcode = "2021ApJ...907L..30A"
    description = "Evidence of 200 TeV photons from HAWC J1825-134"
    source_object_class = SourceCatalogObjectExtraHAWC

    def __init__(
        self,
        filename=(
            "$FEUPY_DATA/dedicated_publications/hawc/2021ApJ...907L..30A/hawc_2021_catalog.ecsv"
        ),
    ):
        """Initialize the dedicated HAWC catalog.

        Parameters
        ----------
        filename : str or `~pathlib.Path`, optional
            Path to the catalog ECSV file.
        """
        table = Table.read(make_path(filename), format="ascii.ecsv")
        super().__init__(table=table, source_name_key="source_name")
