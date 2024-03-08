#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gammapy.irf import load_irf_dict_from_file
from gammapy.data import observatory_locations
from astropy.coordinates import Angle
from astropy.units import Quantity

from astropy import units as u


# In[2]:


__all__ = [
    "Irfs"
]


# In[3]:


_IRFS_OPTIONS = [
    ['South', 'AverageAz', '20deg', '0.5h'],
     ['South', 'AverageAz', '20deg', '5h'],
     ['South', 'AverageAz', '20deg', '50h'],
     ['South', 'NorthAz', '20deg', '0.5h'],
     ['South', 'NorthAz', '20deg', '5h'],
     ['South', 'NorthAz', '20deg', '50h'],
     ['South', 'SouthAz', '20deg', '0.5h'],
     ['South', 'SouthAz', '20deg', '5h'],
     ['South', 'SouthAz', '20deg', '50h'],
     ['South', 'AverageAz', '40deg', '0.5h'],
     ['South', 'AverageAz', '40deg', '5h'],
     ['South', 'AverageAz', '40deg', '50h'],
     ['South', 'NorthAz', '40deg', '0.5h'],
     ['South', 'NorthAz', '40deg', '5h'],
     ['South', 'NorthAz', '40deg', '50h'],
     ['South', 'SouthAz', '40deg', '0.5h'],
     ['South', 'SouthAz', '40deg', '5h'],
     ['South', 'SouthAz', '40deg', '50h'],
     ['South', 'AverageAz', '60deg', '0.5h'],
     ['South', 'AverageAz', '60deg', '5h'],
     ['South', 'AverageAz', '60deg', '50h'],
     ['South', 'NorthAz', '60deg', '0.5h'],
     ['South', 'NorthAz', '60deg', '5h'],
     ['South', 'NorthAz', '60deg', '50h'],
     ['South', 'SouthAz', '60deg', '0.5h'],
     ['South', 'SouthAz', '60deg', '5h'],
     ['South', 'SouthAz', '60deg', '50h'],
     ['South-SSTSubArray', 'AverageAz', '20deg', '0.5h'],
     ['South-SSTSubArray', 'AverageAz', '20deg', '5h'],
     ['South-SSTSubArray', 'AverageAz', '20deg', '50h'],
     ['South-SSTSubArray', 'NorthAz', '20deg', '0.5h'],
     ['South-SSTSubArray', 'NorthAz', '20deg', '5h'],
     ['South-SSTSubArray', 'NorthAz', '20deg', '50h'],
     ['South-SSTSubArray', 'SouthAz', '20deg', '0.5h'],
     ['South-SSTSubArray', 'SouthAz', '20deg', '5h'],
     ['South-SSTSubArray', 'SouthAz', '20deg', '50h'],
     ['South-SSTSubArray', 'AverageAz', '40deg', '0.5h'],
     ['South-SSTSubArray', 'AverageAz', '40deg', '5h'],
     ['South-SSTSubArray', 'AverageAz', '40deg', '50h'],
     ['South-SSTSubArray', 'NorthAz', '40deg', '0.5h'],
     ['South-SSTSubArray', 'NorthAz', '40deg', '5h'],
     ['South-SSTSubArray', 'NorthAz', '40deg', '50h'],
     ['South-SSTSubArray', 'SouthAz', '40deg', '0.5h'],
     ['South-SSTSubArray', 'SouthAz', '40deg', '5h'],
     ['South-SSTSubArray', 'SouthAz', '40deg', '50h'],
     ['South-SSTSubArray', 'AverageAz', '60deg', '0.5h'],
     ['South-SSTSubArray', 'AverageAz', '60deg', '5h'],
     ['South-SSTSubArray', 'AverageAz', '60deg', '50h'],
     ['South-SSTSubArray', 'NorthAz', '60deg', '0.5h'],
     ['South-SSTSubArray', 'NorthAz', '60deg', '5h'],
     ['South-SSTSubArray', 'NorthAz', '60deg', '50h'],
     ['South-SSTSubArray', 'SouthAz', '60deg', '0.5h'],
     ['South-SSTSubArray', 'SouthAz', '60deg', '5h'],
     ['South-SSTSubArray', 'SouthAz', '60deg', '50h'],
     ['South-MSTSubArray', 'AverageAz', '20deg', '0.5h'],
     ['South-MSTSubArray', 'AverageAz', '20deg', '5h'],
     ['South-MSTSubArray', 'AverageAz', '20deg', '50h'],
     ['South-MSTSubArray', 'NorthAz', '20deg', '0.5h'],
     ['South-MSTSubArray', 'NorthAz', '20deg', '5h'],
     ['South-MSTSubArray', 'NorthAz', '20deg', '50h'],
     ['South-MSTSubArray', 'SouthAz', '20deg', '0.5h'],
     ['South-MSTSubArray', 'SouthAz', '20deg', '5h'],
     ['South-MSTSubArray', 'SouthAz', '20deg', '50h'],
     ['South-MSTSubArray', 'AverageAz', '40deg', '0.5h'],
     ['South-MSTSubArray', 'AverageAz', '40deg', '5h'],
     ['South-MSTSubArray', 'AverageAz', '40deg', '50h'],
     ['South-MSTSubArray', 'NorthAz', '40deg', '0.5h'],
     ['South-MSTSubArray', 'NorthAz', '40deg', '5h'],
     ['South-MSTSubArray', 'NorthAz', '40deg', '50h'],
     ['South-MSTSubArray', 'SouthAz', '40deg', '0.5h'],
     ['South-MSTSubArray', 'SouthAz', '40deg', '5h'],
     ['South-MSTSubArray', 'SouthAz', '40deg', '50h'],
     ['South-MSTSubArray', 'AverageAz', '60deg', '0.5h'],
     ['South-MSTSubArray', 'AverageAz', '60deg', '5h'],
     ['South-MSTSubArray', 'AverageAz', '60deg', '50h'],
     ['South-MSTSubArray', 'NorthAz', '60deg', '0.5h'],
     ['South-MSTSubArray', 'NorthAz', '60deg', '5h'],
     ['South-MSTSubArray', 'NorthAz', '60deg', '50h'],
     ['South-MSTSubArray', 'SouthAz', '60deg', '0.5h'],
     ['South-MSTSubArray', 'SouthAz', '60deg', '5h'],
     ['South-MSTSubArray', 'SouthAz', '60deg', '50h'],
     ['North', 'AverageAz', '20deg', '0.5h'],
     ['North', 'AverageAz', '20deg', '5h'],
     ['North', 'AverageAz', '20deg', '50h'],
     ['North', 'NorthAz', '20deg', '0.5h'],
     ['North', 'NorthAz', '20deg', '5h'],
     ['North', 'NorthAz', '20deg', '50h'],
     ['North', 'SouthAz', '20deg', '0.5h'],
     ['North', 'SouthAz', '20deg', '5h'],
     ['North', 'SouthAz', '20deg', '50h'],
     ['North', 'AverageAz', '40deg', '0.5h'],
     ['North', 'AverageAz', '40deg', '5h'],
     ['North', 'AverageAz', '40deg', '50h'],
     ['North', 'NorthAz', '40deg', '0.5h'],
     ['North', 'NorthAz', '40deg', '5h'],
     ['North', 'NorthAz', '40deg', '50h'],
     ['North', 'SouthAz', '40deg', '0.5h'],
     ['North', 'SouthAz', '40deg', '5h'],
     ['North', 'SouthAz', '40deg', '50h'],
     ['North', 'AverageAz', '60deg', '0.5h'],
     ['North', 'AverageAz', '60deg', '5h'],
     ['North', 'AverageAz', '60deg', '50h'],
     ['North', 'NorthAz', '60deg', '0.5h'],
     ['North', 'NorthAz', '60deg', '5h'],
     ['North', 'NorthAz', '60deg', '50h'],
     ['North', 'SouthAz', '60deg', '0.5h'],
     ['North', 'SouthAz', '60deg', '5h'],
     ['North', 'SouthAz', '60deg', '50h'],
     ['North-MSTSubArray', 'AverageAz', '20deg', '0.5h'],
     ['North-MSTSubArray', 'AverageAz', '20deg', '5h'],
     ['North-MSTSubArray', 'AverageAz', '20deg', '50h'],
     ['North-MSTSubArray', 'NorthAz', '20deg', '0.5h'],
     ['North-MSTSubArray', 'NorthAz', '20deg', '5h'],
     ['North-MSTSubArray', 'NorthAz', '20deg', '50h'],
     ['North-MSTSubArray', 'SouthAz', '20deg', '0.5h'],
     ['North-MSTSubArray', 'SouthAz', '20deg', '5h'],
     ['North-MSTSubArray', 'SouthAz', '20deg', '50h'],
     ['North-MSTSubArray', 'AverageAz', '40deg', '0.5h'],
     ['North-MSTSubArray', 'AverageAz', '40deg', '5h'],
     ['North-MSTSubArray', 'AverageAz', '40deg', '50h'],
     ['North-MSTSubArray', 'NorthAz', '40deg', '0.5h'],
     ['North-MSTSubArray', 'NorthAz', '40deg', '5h'],
     ['North-MSTSubArray', 'NorthAz', '40deg', '50h'],
     ['North-MSTSubArray', 'SouthAz', '40deg', '0.5h'],
     ['North-MSTSubArray', 'SouthAz', '40deg', '5h'],
     ['North-MSTSubArray', 'SouthAz', '40deg', '50h'],
     ['North-MSTSubArray', 'AverageAz', '60deg', '0.5h'],
     ['North-MSTSubArray', 'AverageAz', '60deg', '5h'],
     ['North-MSTSubArray', 'AverageAz', '60deg', '50h'],
     ['North-MSTSubArray', 'NorthAz', '60deg', '0.5h'],
     ['North-MSTSubArray', 'NorthAz', '60deg', '5h'],
     ['North-MSTSubArray', 'NorthAz', '60deg', '50h'],
     ['North-MSTSubArray', 'SouthAz', '60deg', '0.5h'],
     ['North-MSTSubArray', 'SouthAz', '60deg', '5h'],
     ['North-MSTSubArray', 'SouthAz', '60deg', '50h'],
     ['North-LSTSubArray', 'AverageAz', '20deg', '0.5h'],
     ['North-LSTSubArray', 'AverageAz', '20deg', '5h'],
     ['North-LSTSubArray', 'AverageAz', '20deg', '50h'],
     ['North-LSTSubArray', 'NorthAz', '20deg', '0.5h'],
     ['North-LSTSubArray', 'NorthAz', '20deg', '5h'],
     ['North-LSTSubArray', 'NorthAz', '20deg', '50h'],
     ['North-LSTSubArray', 'SouthAz', '20deg', '0.5h'],
     ['North-LSTSubArray', 'SouthAz', '20deg', '5h'],
     ['North-LSTSubArray', 'SouthAz', '20deg', '50h'],
     ['North-LSTSubArray', 'AverageAz', '40deg', '0.5h'],
     ['North-LSTSubArray', 'AverageAz', '40deg', '5h'],
     ['North-LSTSubArray', 'AverageAz', '40deg', '50h'],
     ['North-LSTSubArray', 'NorthAz', '40deg', '0.5h'],
     ['North-LSTSubArray', 'NorthAz', '40deg', '5h'],
     ['North-LSTSubArray', 'NorthAz', '40deg', '50h'],
     ['North-LSTSubArray', 'SouthAz', '40deg', '0.5h'],
     ['North-LSTSubArray', 'SouthAz', '40deg', '5h'],
     ['North-LSTSubArray', 'SouthAz', '40deg', '50h'],
     ['North-LSTSubArray', 'AverageAz', '60deg', '0.5h'],
     ['North-LSTSubArray', 'AverageAz', '60deg', '5h'],
     ['North-LSTSubArray', 'AverageAz', '60deg', '50h'],
     ['North-LSTSubArray', 'NorthAz', '60deg', '0.5h'],
     ['North-LSTSubArray', 'NorthAz', '60deg', '5h'],
     ['North-LSTSubArray', 'NorthAz', '60deg', '50h'],
     ['North-LSTSubArray', 'SouthAz', '60deg', '0.5h'],
     ['North-LSTSubArray', 'SouthAz', '60deg', '5h'],
     ['North-LSTSubArray', 'SouthAz', '60deg', '50h']
]


# In[5]:


class Irfs:
    
    IRFS_OPTIONS = _IRFS_OPTIONS
    IRF_version = "prod5 v0.1"
    
    _SITE_ARRAY = {
            'South': '14MSTs37SSTs', 
            'South-SSTSubArray': '37SSTs', 
            'South-MSTSubArray': '14MSTs', 
            'North': '4LSTs09MSTs', 
            'North-MSTSubArray': '09MSTs',
            'North-LSTSubArray': '4LSTs',
        }
    _OBS_TIME = {'0.5h': '1800s', 
        '5h': '18000s',
        '50h':'180000s', }
    
    _DIR_FITS = '$PYTHONPATH/data/irfs/cta-prod5-zenodo-v0.1/fits/'

    def __init__(self):
        self.irfs = irfs
        self.irfs_label = irfs_label
        self.obs_loc = obs_loc

    @property
    def irfs(self):
        return self._irfs

    @irfs.setter
    def irfs(self, irfs):
        if irfs:
            self._irfs = irfs
            
    @property
    def irfs_label(self):
        return self._irfs_label

    @irfs_label.setter
    def irfs_label(self, irfs_label):
        if irfs_label:
            self._irfs_label = irfs_label
            
    @property
    def obs_loc(self):
        return self._obs_loc

    @obs_loc.setter
    def obs_loc(self, obs_loc):
        if obs_loc:
            self._obs_loc = obs_loc
            
    @classmethod
    def get_irfs(cls, irfs_opt):
        dir_FITS = f'CTA-Performance-prod5-v0.1-{irfs_opt[0]}-{irfs_opt[2]}.FITS/'
        isite = irfs_opt[0].rstrip('-SSTSubArray').rstrip('-MSTSubArray').rstrip('-LSTSubArray')
        irfs_file_name = f'Prod5-{isite}-{irfs_opt[2]}-{irfs_opt[1]}-{cls._SITE_ARRAY[irfs_opt[0]]}.{cls._OBS_TIME [irfs_opt[3]]}-v0.1.fits.gz'
        file_name = f'{cls._DIR_FITS}{dir_FITS}{irfs_file_name}'
        Irfs.irfs = load_irf_dict_from_file(file_name)
        Irfs.irfs_label = Irfs.get_irfs_label(irfs_opt)
        Irfs.obs_loc = Irfs.get_obs_loc(irfs_opt)
        return Irfs.irfs

    @staticmethod
    def get_irfs_label(irfs_opt):
        _irfs_opt = ""
        if irfs_opt[1] != 'AverageAz':
            _irfs_opt = f'-{irfs_opt[1]}'
        return  irfs_opt[0] + _irfs_opt + ' (' + irfs_opt[2] + '-' + irfs_opt[3]  + ')'
    
    @staticmethod
    def get_obs_loc(irfs_opt):
        if 'South' in irfs_opt[0]:
            return observatory_locations['cta_south']
        else: return observatory_locations['cta_north'] 

        




