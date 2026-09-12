# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""H.E.S.S. catalog and source classes."""

from astropy.table import Table
from gammapy.catalog.core import SourceCatalog, SourceCatalogObject
from gammapy.estimators import FluxPoints
from gammapy.modeling.models import Models, SkyModel
from gammapy.utils.scripts import make_path

from feupy.utils.formatting import string_to_filename

__all__ = [
    "SourceCatalogObjectExtraHESS",
    "SourceCatalogExtraHESS",
]


class SourceCatalogObjectExtraHESS(SourceCatalogObject):
    """One source from the dedicated H.E.S.S. catalog."""

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
                "$FEUPY_DATA/dedicated_publications/hess/"
                "2019Apercent26A...621A.116H/models.yaml"
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

        return SkyModel(
            spectral_model=spectral_model,
            name=self.name,
        )

    @property
    def flux_points(self):
        """Return source flux points."""
        filename = (
            "$FEUPY_DATA/dedicated_publications/hess/"
            "2019Apercent26A...621A.116H/"
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


class SourceCatalogExtraHESS(SourceCatalog):
    """Catalog for the H.E.S.S. J1825-137 dedicated publication.

    References
    ----------
    https://www.aanda.org/articles/aa/full_html/2019/01/aa34335-18/aa34335-18.html
    """

    tag = "hess-2019A&A"
    bibcode = "2019A&A...621A.116H"
    description = "Particle transport within the pulsar wind nebula HESS J1825-137"

    source_object_class = SourceCatalogObjectExtraHESS

    def __init__(
        self,
        filename=(
            "$FEUPY_DATA/dedicated_publications/hess/"
            "2019Apercent26A...621A.116H/hess_2019_catalog.ecsv"
        ),
    ):
        """Initialize the dedicated H.E.S.S. catalog.

        Parameters
        ----------
        filename : str or `~pathlib.Path`, optional
            Path to the catalog ECSV file.
        """
        table = Table.read(make_path(filename), format="ascii.ecsv")
        super().__init__(
            table=table,
            source_name_key="source_name",
        )
