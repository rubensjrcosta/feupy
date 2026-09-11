# Licensed under a 3-clause BSD style license - see LICENSE
"""Statistical utilities for likelihood and significance analysis."""

from gammapy.stats import WStatCountsStatistic
from scipy.stats import chi2, norm

__all__ = [
    "sigma_to_ts",
    "ts_to_sigma",
    "compute_wstat",
    "compute_significance",
]


def sigma_to_ts(sigma, df=1):
    """Convert Gaussian significance to test statistic.

    Parameters
    ----------
    sigma : float or array-like
        Gaussian significance.
    df : int, optional
        Number of degrees of freedom. Default is 1.

    Returns
    -------
    ts : float or array-like
        Test statistic corresponding to the input significance.
    """
    p_value = 2 * norm.sf(sigma)
    return chi2.isf(p_value, df=df)


def ts_to_sigma(ts, df=1):
    """Convert test statistic to Gaussian significance.

    Parameters
    ----------
    ts : float or array-like
        Test statistic.
    df : int, optional
        Number of degrees of freedom. Default is 1.

    Returns
    -------
    sigma : float or array-like
        Gaussian significance corresponding to the input test statistic.
    """
    p_value = chi2.sf(ts, df=df)
    return norm.isf(0.5 * p_value)


def compute_wstat(n_on, n_off, alpha=0.2):
    """Create a WStat counts statistic.

    Parameters
    ----------
    n_on : int, float, or array-like
        Number of counts in the on region.
    n_off : int, float, or array-like
        Number of counts in the off region.
    alpha : float or array-like, optional
        On/off acceptance ratio. Default is 0.2.

    Returns
    -------
    statistic : `~gammapy.stats.WStatCountsStatistic`
        WStat counts statistic.
    """
    return WStatCountsStatistic(n_on=n_on, n_off=n_off, alpha=alpha)


def compute_significance(dataset_onoff, alpha=0.2):
    """Compute the significance of an on/off dataset.

    Parameters
    ----------
    dataset_onoff : `~gammapy.datasets.SpectrumDatasetOnOff`
        On/off dataset.
    alpha : float, optional
        On/off acceptance ratio. Default is 0.2.

    Returns
    -------
    significance : float
        Square root of the test statistic.
    """
    n_on = dataset_onoff.counts.data.sum()
    n_off = dataset_onoff.counts_off.data.sum()

    return WStatCountsStatistic(
        n_on=n_on,
        n_off=n_off,
        alpha=alpha,
    ).sqrt_ts
