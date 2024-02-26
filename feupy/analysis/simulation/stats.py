#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""stats - Statistics."""


# In[1]:


from scipy.stats import chi2, norm
from gammapy.stats import WStatCountsStatistic
from feupy.utils.scripts import is_documented_by


# In[2]:


__all__ = [
    "StatisticalUtilityFunctions",
#     "set_pointing",
#     "create_observation",
#     "define_on_region",
#     "define_geometry"
]


# In[ ]:


class StatisticalUtilityFunctions:
    """Statistical utility functions
    
    See: https://docs.gammapy.org/1.1/user-guide/stats/
    
    StatisticalUtilityFunctions is represented by `~feupy.analysis.simulation.stats`.
    """    
    
    def __init__(self):
        pass
        
#         self.irfs_list = None
#         self.irfs_label_list = None
    
    @classmethod
    def sigma_to_ts(cls, sigma, df=1):
        """Convert sigma to delta ts"""
        p_value = 2 * norm.sf(sigma)
        return chi2.isf(p_value, df=df)
    
    @classmethod
    def ts_to_sigma(cls, ts, df=1):
        """Convert delta ts to sigma"""
        p_value = chi2.sf(ts, df=df)
        return norm.isf(0.5 * p_value)
    
    @classmethod
    def calculate_sensitivity_lima(cls, n_on_events, n_background, alpha, n_bins_energy,
                               n_bins_gammaness, n_bins_theta2):
        """
        Sensitivity calculation using the Li & Ma formula
        eq. 17 of Li & Ma (1983).
        https://ui.adsabs.harvard.edu/abs/1983ApJ...272..317L/abstract

        We calculate the sensitivity in bins of energy, gammaness and
        theta2

        Parameters
        ---------
        n_on_events:   `numpy.ndarray` number of ON events in the signal region
        n_background:   `numpy.ndarray` number of events in the background region
        alpha: `float` inverse of the number of off positions
        n_bins_energy: `int` number of bins in energy
        n_bins_gammaness: `int` number of bins in gammaness
        n_bins_theta2: `int` number of bins in theta2

        Returns
        ---------
        sensitivity: `numpy.ndarray` sensitivity in percentage of Crab units
        n_excesses_5sigma: `numpy.ndarray` number of excesses corresponding to 
                    a 5 sigma significance

        """

        stat = WStatCountsStatistic(n_on=n_on_events,
                                    n_off=n_background,
                                    alpha=alpha)

        n_excesses_5sigma = stat.excess_matching_significance(5)

        for i in range(0, n_bins_energy):
            for j in range(0, n_bins_gammaness):
                for k in range(0, n_bins_theta2):
                    if n_excesses_5sigma[i][j][k] < 10:
                        n_excesses_5sigma[i][j][k] = 10

                    if n_excesses_5sigma[i, j,
                                         k] < 0.05 * n_background[i][j][k] / 5:
                        n_excesses_5sigma[i, j,
                                          k] = 0.05 * n_background[i][j][k] / 5

        sensitivity = n_excesses_5sigma / n_on_events * 100  # percentage of Crab

        return n_excesses_5sigma, sensitivity

    
    @classmethod
    def calculate_sensitivity_lima_ebin(cls, n_excesses, n_background, alpha, n_bins_energy):
        """
        Sensitivity calculation using the Li & Ma formula
        eq. 17 of Li & Ma (1983).
        https://ui.adsabs.harvard.edu/abs/1983ApJ...272..317L/abstract

        Parameters
        ---------
        n_excesses:   `numpy.ndarray` number of excess events in the signal region
        n_background: `numpy.ndarray` number of events in the background region
        alpha:        `float` inverse of the number of off positions
        n_bins_energy:`int` number of bins in energy

        Returns
        ---------
        sensitivity: `numpy.ndarray` sensitivity in percentage of Crab units
        n_excesses_5sigma: `numpy.ndarray` number of excesses corresponding to 
                    a 5 sigma significance

        """

        if any(len(a) != n_bins_energy for a in (n_excesses, n_background, alpha)):
            raise ValueError(
                'Excess, background and alpha arrays must have the same length')

        stat = WStatCountsStatistic(
            n_on=np.ones_like(n_background),
            n_off=n_background,
            alpha=alpha)

        n_excesses_5sigma = stat.excess_matching_significance(5)

        for i in range(0, n_bins_energy):
            # If the excess needed to get 5 sigma is less than 10,
            # we force it to be at least 10
            if n_excesses_5sigma[i] < 10:
                n_excesses_5sigma[i] = 10
            # If the excess needed to get 5 sigma is less than 5%
            # of the background, we force it to be at least 5% of
            # the background
            if n_excesses_5sigma[i] < 0.05 * n_background[i] * alpha[i]:
                n_excesses_5sigma[i] = 0.05 * n_background[i] * alpha[i]

        sensitivity = n_excesses_5sigma / n_excesses * 100  # percentage of Crab

        return n_excesses_5sigma, sensitivity
    
    @classmethod
    def compute_significance(cls, dataset_onoff, alpha=0.2):
        # Class to compute statistics for Poisson distributed variable with unknown background.
        return WStatCountsStatistic(
            n_on=sum(dataset_onoff.counts.data), 
            n_off=sum(dataset_onoff.counts_off.data), 
            alpha=alpha).sqrt_ts
    
    @classmethod
    @is_documented_by(WStatCountsStatistic)
    def compute_wstat(cls, dataset_onoff, alpha=0.2):
        # Class to compute statistics for Poisson distributed variable with unknown background.
        return WStatCountsStatistic(
            n_on=sum(dataset_onoff.counts.data), 
            n_off=sum(dataset_onoff.counts_off.data), 
            alpha=alpha)


# In[ ]:




