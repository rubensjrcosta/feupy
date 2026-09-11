# Licensed under a 3-clause BSD style license - see LICENSE
"""Utilities for FeuPy datasets and flux-point datasets."""

import os
from pathlib import Path

import numpy as np
from astropy import units as u
from gammapy.datasets import Datasets, FluxPointsDataset
from gammapy.estimators import FluxPoints
from gammapy.modeling.models import SkyModel

__all__ = [
    "get_feupy_data_path",
    "datasets_select_by_name",
    "get_energy_bounds_from_datasets",
    "cut_energy_flux_points_datasets",
    "flux_points_dataset_from_table",
]


def get_feupy_data_path():
    """Return the FeuPy dataset base path.

    The path is read from the ``FEUPY_DATA`` environment variable.

    Returns
    -------
    path : `~pathlib.Path`
        Resolved path to the FeuPy datasets directory.

    Raises
    ------
    RuntimeError
        If ``FEUPY_DATA`` is not defined or the path does not exist.
    """
    if "FEUPY_DATA" not in os.environ:
        raise RuntimeError(
            "FEUPY_DATA environment variable not set.\n"
            "Install feupy-datasets or define FEUPY_DATA path."
        )

    path = Path(os.environ["FEUPY_DATA"]).expanduser().resolve()

    if not path.exists():
        raise RuntimeError(
            f"FEUPY_DATA path does not exist:\n{path}\nCheck dataset installation."
        )

    return path


def datasets_select_by_name(datasets, names):
    """Select datasets by name.

    Parameters
    ----------
    datasets : `~gammapy.datasets.Datasets`
        Collection of datasets.
    names : list of str
        Names of the datasets to select.

    Returns
    -------
    selected : `~gammapy.datasets.Datasets`
        Collection containing only the selected datasets.
    """
    names = set(names)
    return Datasets([dataset for dataset in datasets if dataset.name in names])


def get_energy_bounds_from_datasets(datasets):
    """Calculate minimum and maximum energies from one or more datasets.

    Parameters
    ----------
    datasets : `~gammapy.datasets.FluxPointsDataset` or `~gammapy.datasets.Datasets`
        Dataset or collection of datasets containing spectral energy bounds.

    Returns
    -------
    energy_bounds : `~astropy.units.Quantity`
        Two-element array containing the minimum and maximum energies.
        If both values are equal, a small range is created around the value.
    """
    if isinstance(datasets, FluxPointsDataset):
        datasets = [datasets]

    energy_mins = u.Quantity(
        [min(dataset.data.energy_min).to(u.TeV) for dataset in datasets]
    )
    energy_maxs = u.Quantity(
        [max(dataset.data.energy_max).to(u.TeV) for dataset in datasets]
    )

    energy_min = energy_mins.min()
    energy_max = energy_maxs.max()

    if energy_max == energy_min:
        print(
            "\nThe minimum and maximum energies are equal, "
            "so a small range will be created around the value: "
            f"{energy_max}"
        )
        energy_max += 1 * energy_max.unit
        energy_min = energy_max - 1 * energy_max.unit

    return u.Quantity([energy_min, energy_max])


def cut_energy_flux_points_datasets(
    datasets,
    e_ref_min=None,
    e_ref_max=None,
):
    """Cut flux-point datasets within specified energy limits.

    Parameters
    ----------
    datasets : `~gammapy.datasets.FluxPointsDataset` or `~gammapy.datasets.Datasets`
        Dataset or collection of datasets to filter by energy.
    e_ref_min : `~astropy.units.Quantity`, optional
        Minimum reference energy, inclusive.
    e_ref_max : `~astropy.units.Quantity`, optional
        Maximum reference energy, inclusive.

    Returns
    -------
    filtered : `~gammapy.datasets.FluxPointsDataset` or `~gammapy.datasets.Datasets`
        Dataset or collection containing flux points inside the requested
        energy range.
    """
    single_dataset = False
    if isinstance(datasets, FluxPointsDataset):
        datasets = [datasets]
        single_dataset = True

    filtered_datasets = Datasets()

    for dataset in datasets:
        try:
            if not isinstance(dataset, FluxPointsDataset):
                print(f"Skipping dataset '{dataset.name}' - not a FluxPointsDataset.")
                continue

            flux_points = dataset.data
            models = dataset.models[0] if dataset.models else None
            dataset_name = dataset.name

            if e_ref_min is not None:
                mask_energy = np.array(
                    [e_ref >= e_ref_min for e_ref in flux_points.energy_ref]
                )
                flux_points = FluxPoints.from_table(flux_points.to_table()[mask_energy])

            if e_ref_max is not None:
                mask_energy = np.array(
                    [e_ref <= e_ref_max for e_ref in flux_points.energy_ref]
                )
                flux_points = FluxPoints.from_table(flux_points.to_table()[mask_energy])

            filtered_dataset = FluxPointsDataset(
                models=models,
                data=flux_points,
                name=dataset_name,
            )
            filtered_datasets.append(filtered_dataset)

        except Exception as error:
            print(
                f"\nUnable to cut {dataset.name} FluxPointsDataset."
                f"An error has occurred: {error}."
            )

    if single_dataset:
        return filtered_datasets[0]

    return filtered_datasets


def flux_points_dataset_from_table(
    table,
    reference_model=None,
    sed_type=None,
    name=None,
    kwargs_fp=None,
    kwargs_ds=None,
    model_name=None,
):
    """Create a flux-points dataset from a table.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table containing the flux points.
    reference_model : `~gammapy.modeling.models.SpectralModel`, optional
        Reference spectral model.
    sed_type : str, optional
        Spectral energy distribution type.
    name : str, optional
        Dataset name.
    kwargs_fp : dict, optional
        Additional arguments passed to `~gammapy.estimators.FluxPoints`.
    kwargs_ds : dict, optional
        Additional arguments passed to
        `~gammapy.datasets.FluxPointsDataset`.
    model_name : str, optional
        Model name used when ``reference_model`` is provided.

    Returns
    -------
    dataset : `~gammapy.datasets.FluxPointsDataset`
        Dataset created from the input table.
    """
    if kwargs_fp is None:
        kwargs_fp = {
            "format": "gadf-sed",
            "gti": None,
        }

    if kwargs_ds is None:
        kwargs_ds = {
            "mask_fit": None,
            "mask_safe": None,
            "meta_table": None,
        }

    flux_points = FluxPoints.from_table(
        table=table,
        reference_model=reference_model,
        sed_type=sed_type,
        **kwargs_fp,
    )

    models = None
    if reference_model:
        models = SkyModel(
            spectral_model=reference_model,
            name=model_name or name,
        )

    return FluxPointsDataset(
        models=models,
        data=flux_points,
        name=name,
        **kwargs_ds,
    )
