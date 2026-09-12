# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities to export Gammapy spectral data to Naima."""

from astropy.table import Table

__all__ = [
    "REQUIRED_NAIMA_COLUMNS_NAMES",
    "REQUIRED_NAIMA_COLUMNS",
    "make_naima_tables",
]


REQUIRED_NAIMA_COLUMNS_NAMES = {
    "e_ref": "energy",
    "e_min": "energy_error_lo",
    "e_max": "energy_error_hi",
    "dnde": "flux",
    "dnde_err": "flux_error",
    "dnde_errp": "flux_error_hi",
    "dnde_errn": "flux_error_lo",
    "dnde_ul": "flux_ul",
    "e2dnde": "flux",
    "e2dnde_err": "flux_error",
    "e2dnde_errp": "flux_error_hi",
    "e2dnde_errn": "flux_error_lo",
    "e2dnde_ul": "flux_ul",
    "is_ul": "ul",
}


REQUIRED_NAIMA_COLUMNS = {
    "dnde": [
        "e_ref",
        "dnde",
        "dnde_err",
        "dnde_errp",
        "dnde_errn",
        "dnde_ul",
        "is_ul",
    ],
    "e2dnde": [
        "e_ref",
        "e2dnde",
        "e2dnde_err",
        "e2dnde_errp",
        "e2dnde_errn",
        "e2dnde_ul",
        "is_ul",
    ],
}


def make_naima_tables(datasets, sed_type="dnde"):
    """Convert Gammapy datasets to Naima-compatible tables.

    Parameters
    ----------
    datasets : iterable
        Gammapy datasets containing spectral data, typically
        `~gammapy.datasets.FluxPointsDataset` objects.
    sed_type : {"dnde", "e2dnde"}, optional
        Spectral representation to export.

    Returns
    -------
    tables : list of `~astropy.table.Table`
        Tables formatted for Naima likelihood fitting.

    Raises
    ------
    ValueError
        If ``sed_type`` is not supported.
    """
    if sed_type not in REQUIRED_NAIMA_COLUMNS:
        valid = ", ".join(REQUIRED_NAIMA_COLUMNS)
        raise ValueError(f"Invalid sed_type '{sed_type}'. Available options: {valid}.")

    tables = []

    for dataset in datasets:
        data = dataset.data.to_table(sed_type=sed_type)
        table = Table(meta={"name": dataset.name})

        for column in REQUIRED_NAIMA_COLUMNS[sed_type]:
            if column in data.colnames:
                column_name = REQUIRED_NAIMA_COLUMNS_NAMES[column]
                table[column_name] = data[column]

        tables.append(table)

    return tables
