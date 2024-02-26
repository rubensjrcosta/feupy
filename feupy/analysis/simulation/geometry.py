#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Map geometry."""


# In[7]:





# In[4]:


from astropy import units as u
from astropy.units import Quantity


# In[5]:


__all__ = [
    "GeometryParameters"
]


# In[ ]:





# In[6]:


class GeometryParameters:
    """Container for geometry parameters.

    Parameters
    ----------  
    e_reco_min :  `~astropy.units.Quantity`
        Minimal energy for simulation
    e_reco_max : `~astropy.units.Quantity`
        Maximal energy for simulation
    nbin_reco : int
    e_true_min :  `~astropy.units.Quantity`
        Minimal energy for simulation
    e_true_max : `~astropy.units.Quantity`
        Maximal energy for simulation
    nbin_true : int
    """
    @u.quantity_input(
        e_reco_min=u.eV, 
        e_reco_max=u.eV,
        e_true_min=u.eV, 
        e_true_max=u.eV
    )
    def __init__(self,
                 e_reco_min=None,
                 e_reco_max=None,
                 nbin_reco: int=None,
                 e_true_min=None,
                 e_true_max=None,
                 nbin_true: int=None,
                ):
        self.e_reco_min = Quantity(e_reco_min, "TeV")
        self.e_reco_max = Quantity(e_reco_max, "TeV")
        self.nbin_reco = nbin_reco
        self.e_true_min = Quantity(e_true_min, "TeV")
        self.e_true_max = Quantity(e_true_max, "TeV")
        self.nbin_true = nbin_true

    def __str__(self):
        """Geometry summary report (`str`)."""
        ss = '*** Basic parameters ***\n\n'
        ss += 'e_reco_min={:.2f}\n'.format(self.e_reco_min).replace(' ', '')
        ss += 'e_reco_max={:.2f}\n'.format(self.e_reco_max).replace(' ', '')
        ss += 'nbin_reco={}\n'.format(self.nbin_reco)
        ss += 'e_true_min={:.2f}\n'.format(self.e_true_min).replace(' ', '')
        ss += 'e_true_max={:.2f}\n'.format(self.e_true_max).replace(' ', '')
        ss += 'nbin_true={}\n'.format(self.nbin_true)
        return ss.replace('=', ' = ')


# In[ ]:




