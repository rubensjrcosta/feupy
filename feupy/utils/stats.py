# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Statistical utilities for likelihood and significance analysis.
"""
import numpy as np
from scipy.stats import chi2, norm
from gammapy.stats import WStatCountsStatistic

__all__ = [
    "sigma_to_ts",
    "ts_to_sigma",
    "compute_wstat",
    "compute_significance",
]


def sigma_to_ts(sigma, df=1):
    """Convert Gaussian sigma to Test Statistic (TS)."""
    p_value = 2 * norm.sf(sigma)
    return chi2.isf(p_value, df=df)


def ts_to_sigma(ts, df=1):
    """Convert Test Statistic (TS) to Gaussian sigma."""
    p_value = chi2.sf(ts, df=df)
    return norm.isf(0.5 * p_value)


def compute_wstat(n_on, n_off, alpha=0.2):
    """Compute WStatCountsStatistic."""
    return WStatCountsStatistic(n_on=n_on, n_off=n_off, alpha=alpha)


def compute_significance(dataset_onoff, alpha=0.2):
    """Compute Li & Ma-like significance."""
    return WStatCountsStatistic(
        n_on=sum(dataset_onoff.counts.data),
        n_off=sum(dataset_onoff.counts_off.data),
        alpha=alpha,
    ).sqrt_ts