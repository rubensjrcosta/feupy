# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for the pulsar_spectra catalog."""

import logging

import numpy as np
import yaml
from astropy.table import Column, Table
from astropy.units import Quantity
from gammapy.datasets import Datasets, FluxPointsDataset
from gammapy.estimators import FluxPoints
from gammapy.utils.scripts import make_path

from feupy.utils.conversions import frequency_to_energy, jy_to_erg_cm2_s

log = logging.getLogger(__name__)

__all__ = [
    "read_pulsar_spectra_catalog",
    "create_pulsar_flux_points_table",
    "get_pulsar_flux_points_tables",
    "get_pulsar_flux_points_table",
    "get_pulsar_flux_points_datasets",
]


def read_pulsar_spectra_catalog(filename=None):
    """Read the pulsar_spectra catalog from a YAML file.

    Parameters
    ----------
    filename : str or `~pathlib.Path`, optional
        Path to the pulsar_spectra YAML file.

    Returns
    -------
    catalog : dict
        Pulsar spectra catalog. An empty dictionary is returned if the file
        cannot be found or parsed.
    """
    if filename is None:
        filename = "$FEUPY_DATA/catalogs/pulsar_spectra/pulsar_spectra.yaml"

    try:
        with make_path(filename).open(encoding="utf-8") as yaml_file:
            return yaml.safe_load(yaml_file) or {}
    except yaml.YAMLError as error:
        log.error("YAML loading error: %s", error)
    except FileNotFoundError:
        log.error("Pulsar catalog file not found: %s", filename)

    return {}


def create_pulsar_flux_points_table(pulsar_jname):
    """Create a flux-points table for a pulsar.

    Parameters
    ----------
    pulsar_jname : str
        Pulsar J-name.

    Returns
    -------
    table : `~astropy.table.Table` or None
        Flux-points table, or None if the pulsar is not available.
    """
    catalog = read_pulsar_spectra_catalog()
    metadata = {
        "source_name": f"PSR {pulsar_jname}",
        "pulsar_jname": pulsar_jname,
        "catalog": "pulsar_spectra",
        "sed_type": "e2dnde",
        "comments": ["Reference: https://doi.org/10.1017/pasa.2022.52"],
        "bibcode": "2022PASA...39...56S",
    }

    try:
        freqs, _bands, fluxes, flux_errors, references = catalog[pulsar_jname]

        freqs_mhz = Quantity(freqs, "MHz")
        fluxes_mjy = Quantity(fluxes, "mJy")
        flux_errors_mjy = Quantity(flux_errors, "mJy")

        table = Table(meta=metadata)
        table["ref"] = Column(
            data=np.asarray(references, dtype="U20"),
            description="Reference label",
        )
        table["e_ref"] = Column(
            data=frequency_to_energy(freqs_mhz),
            unit="eV",
            description="Reference energy",
            format=".3e",
        )
        table["e2dnde"] = Column(
            data=jy_to_erg_cm2_s(freqs_mhz, fluxes_mjy),
            unit="erg cm^-2 s^-1",
            description="Differential flux",
            format=".3e",
        )
        table["e2dnde_err"] = Column(
            data=jy_to_erg_cm2_s(freqs_mhz, flux_errors_mjy),
            unit="erg cm^-2 s^-1",
            description="Differential flux uncertainty",
            format=".3e",
        )
        return table
    except KeyError:
        log.error("Pulsar J-name '%s' not found in the catalog.", pulsar_jname)
    except Exception as error:
        log.error("Error processing pulsar '%s': %s", pulsar_jname, error)

    return None


def get_pulsar_flux_points_table(pulsar_jname):
    """Return the flux-points table for a pulsar.

    Parameters
    ----------
    pulsar_jname : str
        Pulsar J-name.

    Returns
    -------
    table : `~astropy.table.Table` or None
        Flux-points table, or None if the pulsar is not available.
    """
    return create_pulsar_flux_points_table(pulsar_jname)


def get_pulsar_flux_points_tables(pulsar_jname):
    """Return flux-points tables grouped by reference.

    Parameters
    ----------
    pulsar_jname : str
        Pulsar J-name.

    Returns
    -------
    tables : list of `~astropy.table.Table`
        Flux-points tables grouped by publication reference.
    """
    table = get_pulsar_flux_points_table(pulsar_jname)
    if table is None:
        return []

    grouped_tables = []
    for group in table.group_by("ref").groups:
        group_table = group[["e_ref", "e2dnde", "e2dnde_err"]]
        group_table.meta.update({"ref": group["ref"][0], **table.meta})
        grouped_tables.append(group_table)

    return grouped_tables


def get_pulsar_flux_points_datasets(pulsar_jname):
    """Return flux-points datasets grouped by reference.

    Parameters
    ----------
    pulsar_jname : str
        Pulsar J-name.

    Returns
    -------
    datasets : `~gammapy.datasets.Datasets`
        Flux-points datasets grouped by publication reference.
    """
    grouped_tables = get_pulsar_flux_points_tables(pulsar_jname)
    if not grouped_tables:
        return Datasets()

    datasets = Datasets()
    for table in grouped_tables:
        label = table.meta["ref"]
        flux_points = FluxPoints.from_table(
            table=table,
            sed_type=table.meta["sed_type"],
        )
        datasets.append(FluxPointsDataset(data=flux_points, name=label))

    return datasets
