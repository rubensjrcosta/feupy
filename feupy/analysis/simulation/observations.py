#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


from astropy.coordinates import Angle
from astropy.units import Quantity

from astropy import units as u
from gammapy.estimators import SensitivityEstimator
from feupy.utils.scripts import is_documented_by
from feupy.utils.datasets import flux_points_dataset_from_table


# In[2]:


__all__ = [
    "ObservationParameters"
]


# In[35]:


class ObservationParameters:
    """Container for observation parameters.

    Parameters
    ----------
    livetime :  `~astropy.units.Quantity`
        Livetime exposure of the simulated observation
    on_region_radius : `~astropy.units.Quantity`
        Integration radius of the ON extraction region
    offset : `~astropy.units.Quantity`
        Pointing position offset    
    n_obs : int
        Number of simulations of each observation   
    alpha : `~astropy.units.Quantity`
        Normalisation between ON and OFF regions
    """
    @u.quantity_input(livetime=u.h, on_region_radius=u.deg, offset=u.deg)
    def __init__(self,
                 livetime=None,
                 on_region_radius=None, 
                 offset=None, 
                 n_obs=None
                ):
        self.livetime = livetime
        self.on_region_radius = on_region_radius
        self.offset = offset
        self.n_obs = n_obs

    @property
    def livetime(self):
        return self._livetime

    @livetime.setter
    def livetime(self, livetime):
        if livetime is not None:
            self._livetime = Quantity(livetime, "h")
        else: self._livetime = livetime

    @property
    def on_region_radius(self):
        return self._on_region_radius

    @on_region_radius.setter
    def on_region_radius(self, on_region_radius):
        if on_region_radius is not None:
            self._on_region_radius = Angle(Quantity(on_region_radius, "deg"))
        else: self._on_region_radius = on_region_radius

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, offset):
        if offset is not None:
            self._offset = Quantity(offset, "deg")
        else: self._offset = offset

    @property
    def n_obs(self):
        return self._n_obs

    @n_obs.setter
    def n_obs(self, n_obs):
        self._n_obs = n_obs
                        
    def __str__(self):
        """Observation summary report (`str`)."""
        ss = '*** Basic parameters ***\n\n'
        if self.livetime is not None:
            ss += 'livetime={:.2f}\n'.format(self.livetime).replace(' ', '')
        else: ss += 'livetime=None\n'
        if self.on_region_radius is not None:
            ss += 'on_region_radius={:.2f}\n'.format(self.on_region_radius).replace(' ', '')
        else: ss += 'on_region_radius=None\n'
        if self.offset is not None:
            ss += 'offset={:.2f}\n'.format(self.offset).replace(' ', '')
        else: ss += 'offset=None\n'
        if self.n_obs is not None:
            ss += 'n_obs={}\n'.format(self.n_obs)
        else: ss += 'n_obs=None\n'
        return ss.replace('=', ' = ')


# In[4]:


@is_documented_by(SensitivityEstimator)
def sensitivity_estimator(
    spectrum=None,
    n_sigma=5.0,
    gamma_min=10,
    bkg_syst_fraction=0.05,
    dataset_onoff=None,
):
    sensitivity = SensitivityEstimator(
        spectrum=spectrum,
        gamma_min=gamma_min, 
        n_sigma=n_sigma, 
        bkg_syst_fraction=bkg_syst_fraction
)
    return sensitivity, sensitivity.run(dataset_onoff)


# In[39]:





# In[ ]:




