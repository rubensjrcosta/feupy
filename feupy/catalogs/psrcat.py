# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""ATNF Pulsar Catalogue and source classes."""

from astropy.table import Table
from gammapy.catalog.core import SourceCatalog, SourceCatalogObject
from gammapy.utils.scripts import make_path

__all__ = [
    "SourceCatalogPSRCAT",
    "SourceCatalogObjectPSRCAT",
]


class SourceCatalogObjectPSRCAT(SourceCatalogObject):
    """One source from the ATNF Pulsar Catalogue.

    References
    ----------
    Manchester, R. N., Hobbs, G. B., Teoh, A. & Hobbs, M. (2005),
    *The Australia Telescope National Facility Pulsar Catalogue*,
    Astronomical Journal, 129, 1993-2006.
    """

    _source_name_key = "NAME"

    def __str__(self):
        return self.info()

    def info(self, info="all"):
        """Return summary information for the source.

        Parameters
        ----------
        info : str, optional
            Comma-separated list of sections to include. Available options are
            ``"basic"``, ``"position"``, ``"timing-profile"``, ``"distance"``,
            ``"associations-survey"``, and ``"derived"``. The default,
            ``"all"``, includes every section.

        Returns
        -------
        info : str
            Formatted source information.
        """
        if info == "all":
            info = "basic,position,timing-profile,distance,associations-survey,derived"

        text = ""
        options = info.split(",")

        if "basic" in options:
            text += self._info_basic()
        if "position" in options:
            text += self._info_position()
        if "timing-profile" in options:
            text += self._info_timing_profile()
        if "distance" in options:
            text += self._info_distance()
        if "associations-survey" in options:
            text += self._info_associations_survey()
        if "derived" in options:
            text += self._info_derived()

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
            f"RA: {self.data.RAJ2000:.3f} ± {self.data.RAJ2000_ERR:.3f}\n"
            f"DEC: {self.data.DEJ2000:.3f} ± {self.data.DEJ2000_ERR:.3f}\n"
        )

    def _info_timing_profile(self):
        """Return timing and profile information."""
        return (
            "\n*** Timing and profile info ***\n\n"
            f"P0: {self.data.P0.value:.3e} ± {self.data.P0_ERR:.3e}\n"
        )

    def _info_distance(self):
        """Return distance information."""
        return (
            "\n*** Distance info ***\n\n"
            f"Dist: {self.data.DIST:.2e}\n"
            f"Dist_DM: {self.data.DIST_DM:.2e}\n"
        )

    def _info_associations_survey(self):
        """Return association and survey information."""
        return (
            "\n*** Associations and survey info ***\n\n"
            f"Assoc: {self.data.ASSOC}\n"
            f"Type: {self.data.TYPE}\n"
        )

    def _info_derived(self):
        """Return derived pulsar parameters."""
        return (
            "\n*** Derived parameters info ***\n\n"
            f"Age: {self.data.AGE:.2e}\n"
            f"BSurf: {self.data.BSURF:.2e}\n"
            f"E_dot: {self.data.EDOT:.2e}\n"
        )


class SourceCatalogPSRCAT(SourceCatalog):
    """ATNF Pulsar Catalogue.

    Each entry is represented by
    `~feupy.catalogs.psrcat.SourceCatalogObjectPSRCAT`.

    References
    ----------
    Manchester, R. N., Hobbs, G. B., Teoh, A. & Hobbs, M. (2005),
    *The Australia Telescope National Facility Pulsar Catalogue*,
    Astronomical Journal, 129, 1993-2006.
    """

    tag = "psrcat"
    bibcode = "2005AJ....129.1993M"
    description = (
        "ATNF Pulsar Catalogue, a comprehensive database of all published pulsars"
    )

    source_object_class = SourceCatalogObjectPSRCAT

    def __init__(
        self,
        filename="$FEUPY_DATA/catalogs/psrcat/psrcat_catalog.fits",
    ):
        """Initialize the ATNF Pulsar Catalogue.

        Parameters
        ----------
        filename : str or `~pathlib.Path`, optional
            Path to the PSRCAT FITS file.
        """
        table = Table.read(make_path(filename), format="fits")
        super().__init__(table=table, source_name_key="NAME")

    @property
    def PSR_PARAMS(self):
        """Pulsar parameter names available in the catalog."""
        return self.table.colnames

    @property
    def PSR_PARAMS_DESCRIPTION(self):
        """Formatted description of pulsar parameters."""
        text = "\n*** The Pulsar Parameters ***\n\n"

        for parameter in self.PSR_PARAMS:
            column = self.table[parameter]
            unit = f" ({column.unit})" if column.unit is not None else ""
            text += f"{column.name}: {column.description}{unit}\n"

        return text
