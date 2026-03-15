# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Gammapy → Naima export utilities.

This module converts spectral information from Gammapy objects
(FluxPointsDataset, SpectrumDataset, etc.) into Astropy Tables
formatted for Naima.

Astropy Tables are used strictly as an export format and are not
part of the Gammapy analysis workflow.
"""

from __future__ import annotations

from astropy.table import Table

__all__ = [
    "REQUIRED_NAIMA_COLUMNS_NAMES",
    "REQUIRED_NAIMA_COLUMNS",
    "make_naima_tables",
]


# ---------------------------------------------------------
# Column name translation (Gammapy → Naima)
# ---------------------------------------------------------

REQUIRED_NAIMA_COLUMNS_NAMES = {

    # Energy
    "e_ref": "energy",
    "e_min": "energy_error_lo",
    "e_max": "energy_error_hi",

    # Differential flux
    "dnde": "flux",
    "dnde_err": "flux_error",
    "dnde_errp": "flux_error_hi",
    "dnde_errn": "flux_error_lo",
    "dnde_ul": "flux_ul",

    # SED flux (E² dN/dE)
    "e2dnde": "flux",
    "e2dnde_err": "flux_error",
    "e2dnde_errp": "flux_error_hi",
    "e2dnde_errn": "flux_error_lo",
    "e2dnde_ul": "flux_ul",

    # Upper limit flag
    "is_ul": "ul",
}


# ---------------------------------------------------------
# Required Gammapy columns per SED type
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Main export function
# ---------------------------------------------------------

def make_naima_tables(datasets, sed_type: str = "dnde"):
    """
    Convert Gammapy datasets into Naima-compatible Astropy tables.

    Parameters
    ----------
    datasets : iterable
        Iterable of Gammapy datasets containing spectral data
        (typically FluxPointsDataset).

    sed_type : {"dnde", "e2dnde"}, optional
        Spectral representation to export.

    Returns
    -------
    tables : list of `astropy.table.Table`
        Tables formatted for Naima likelihood fitting.
    """

    tables = []

    for dataset in datasets:

        data = dataset.data.to_table(sed_type=sed_type)

        table = Table()
        table.meta["name"] = dataset.name

        # Select columns present in dataset
        available_columns = data.colnames
        columns = [
            col for col in REQUIRED_NAIMA_COLUMNS[sed_type]
            if col in available_columns
        ]

        # Rename columns for Naima
        for column in columns:
            column_naima = REQUIRED_NAIMA_COLUMNS_NAMES[column]
            table[column_naima] = data[column]

        tables.append(table)

    return tables