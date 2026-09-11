# Licensed under a 3-clause BSD style license - see LICENSE
"""Akaike Information Criterion utilities."""

import numpy as np

__all__ = ["calculate_aic", "calculate_relative_aic"]


def calculate_aic(datasets, fit_result):
    """Calculate the corrected Akaike Information Criterion (AICc).

    Parameters
    ----------
    datasets : iterable
        Collection of datasets used in the fit. Each dataset is expected to
        provide an ``is_ul`` mask through ``dataset.data.is_ul.data``.
    fit_result : object
        Fit result providing ``success``, ``total_stat``, and ``models``.
        The models object must expose the free parameters through
        ``models.parameters.free_parameters.names``.

    Returns
    -------
    aicc : float
        Corrected Akaike Information Criterion.

    Raises
    ------
    ValueError
        If the fit was not successful or if the number of data points is too
        small for the AICc correction.
    AttributeError
        If ``fit_result.models`` is missing.
    TypeError
        If ``fit_result.models`` does not provide the expected Gammapy-like
        parameter interface.
    """
    if not fit_result.success:
        raise ValueError("Fit was not successful.")

    n_points = sum(int(np.sum(~dataset.data.is_ul.data)) for dataset in datasets)
    total_stat = float(fit_result.total_stat)

    models = getattr(fit_result, "models", None)
    if models is None:
        raise AttributeError("fit_result.models is missing.")

    try:
        n_parameters = len(models.parameters.free_parameters.names)
    except AttributeError as error:
        raise TypeError(
            "fit_result.models must provide models.parameters.free_parameters.names."
        ) from error

    denominator = n_points - n_parameters - 1
    if denominator <= 0:
        raise ValueError(
            "AICc requires the number of data points to satisfy N > k + 1."
        )

    aic = total_stat + 2 * n_parameters
    correction = (2 * n_parameters**2 + 2 * n_parameters) / denominator

    return aic + correction


def calculate_relative_aic(datasets, fit_h0, fit_h1):
    """Calculate the relative AICc difference between two fit hypotheses.

    Parameters
    ----------
    datasets : iterable
        Collection of datasets used in both fits.
    fit_h0 : object
        Fit result for the reference hypothesis.
    fit_h1 : object
        Fit result for the alternative hypothesis.

    Returns
    -------
    relative_difference : float
        Relative AICc difference in percent, defined as
        ``(1 - AICc_H1 / AICc_H0) * 100``.
    """
    aic_h0 = calculate_aic(datasets, fit_h0)
    aic_h1 = calculate_aic(datasets, fit_h1)

    return (1 - aic_h1 / aic_h0) * 100
