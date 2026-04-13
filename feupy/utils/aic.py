# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Akaike Information Criterion utilities.
"""

import numpy as np

__all__ = ["calculate_aic", "calculate_relative_aic"]


def calculate_aic(datasets, fit_result):
    """Compute AIC and corrected AIC."""

    if not fit_result.success:
        raise ValueError("Fit was not successful.")

    N_pt = sum(
        int(np.sum(~dataset.data.is_ul.data))
        for dataset in datasets
    )

    Wstat = float(fit_result.total_stat)
    k = int(len(fit_result.models.parameters.free_parameters.names))

    AIC = Wstat + 2 * k
    AICc = AIC + ((2 * k**2 + 2 * k) / (N_pt - k - 1))

    return AICc


def calculate_relative_aic(datasets, fit_H0, fit_H1):
    """Relative AIC difference in %."""
    aic0 = calculate_aic(datasets, fit_H0)
    aic1 = calculate_aic(datasets, fit_H1)

    return (1 - aic1 / aic0) * 100