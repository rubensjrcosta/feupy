#!/usr/bin/env python
# coding: utf-8

# In[18]:


from gammapy.irf import load_irf_dict_from_file
from gammapy.data import observatory_locations

from astropy import units as u


# In[19]:


__all__ = [
    "CTAOPerf",
    "ObservationParameters"
]


# In[3]:


# class CTAOPerf:
#     """CTAO Instrument Response Functions - prod5 version v0.1
    
#     See: https://zenodo.org/records/5499840#.YUya5WYzbUI
    
#     CTA is represented by `~feupy.cta.CTAOPerf`.
#     """    
#     IRF_ZENITH = [20, 40, 60]
#     IRF_HOURS = [0.5, 5, 50]
#     SITE = ["North", "South"]
#     OBS_LOC = {"North": "cta_north", 
#                "South": "cta_south",}

#     def __init__(self, irf_zenith, irf_hours, site):
        
#         assert self.IRF_ZENITH.count(irf_zenith) > 0, f"Zenith angle {self.IRF_ZENITH} is not in the list: {IRF_ZENITH}!"
#         assert self.IRF_HOURS.count(irf_hours) > 0, f"Observation time {self.IRF_HOURS} is not in the list: {IRF_HOURS}!"
#         assert self.SITE.count(site) > 0, f"Site {site} is not in the list: {self.SITE}!"

#         self.irf_zenith = irf_zenith
#         self.irf_hours = irf_hours
#         self.site = site
#         self.obs_loc = observatory_locations[self.OBS_LOC[site]]
#         self.irf_name = f'{site}_z{irf_zenith}_{irf_hours}h'
#         self.irf = self.load_irf()
    
#     def load_irf(self):
#         dirbasename="$PYTHONPATH/feupy/data/irfs/cta-prod5-zenodo-v0.1/fits/"
        
#         if self.site == 'South':
#             telescopes = '14MSTs37SSTs'
#         else: telescopes = '4LSTs09MSTs'
        
#         dir_irf = f'CTA-Performance-prod5-v0.1-{self.site}-{self.irf_zenith}deg.FITS/'
        
#         if self.irf_hours == 0.5:
#             seconds = 1800
#         elif self.irf_hours == 5:
#             seconds = 18000
#         else: seconds = 180000
            
#         irf_file_name = f'Prod5-{self.site}-{self.irf_zenith}deg-{self.site}Az-{telescopes}.{seconds}s-v0.1.fits.gz'
#         return load_irf_dict_from_file(f'{dirbasename}{dir_irf}{irf_file_name}')


# In[20]:


class CTAOPerf:
    """CTAO Instrument Response Functions - prod5 version v0.1
    
    See: https://zenodo.org/records/5499840#.YUya5WYzbUI
    
    CTAOPerf is represented by `~feupy.cta.CTAOPerf`.
    """    


    def __init__(self):
        self.irfs_list = None
        self.irfs_label_list = None
    
    def get_irfs(self, irfs_label):
        index = self.irfs_label_list.index(irfs_label)
        return self.irfs_list[index]
    
    def get_irfs_label(self, irfs):
        index = self.irfs_list.index(irfs)
        return self.irfs_label_list[index]
    
    def get_obs_loc(self, label):
        if 'South' in label:
            return observatory_locations['cta_south']
        else: return observatory_locations['cta_north'] 
    
    def load_irfs(self):
        dir_fits = '$PYTHONPATH/feupy/data/irfs/cta-prod5-zenodo-v0.1/fits/'
        zenith = ['20deg', '40deg', '60deg']
        site_array = [
            ('South', '14MSTs37SSTs'), 
            ('South-SSTSubArray', '37SSTs'), 
            ('South-MSTSubArray', '14MSTs'), 
            ('North', '4LSTs09MSTs'), 
            ('North-MSTSubArray', '09MSTs'),
            ('North-LSTSubArray', '4LSTs'),
        ]
        opti = ['AverageAz', 'NorthAz', 'SouthAz']
        obs_time = [('1800s', '0.5h'), ('18000s', '5h'), ('180000s', '50h')]
        ctao_irfs_list = []  # will be filled with different performance
        irfs_label_list = []  # will be filled with different performance irfs_label_list for the legend
        for isite_array in site_array: 
            for izenith in zenith:
                dir_FITS = f'CTA-Performance-prod5-v0.1-{isite_array[0]}-{izenith}.FITS/'
                isite = isite_array[0].rstrip('-SSTSubArray').rstrip('-MSTSubArray').rstrip('-LSTSubArray')
                for iopti in opti:
                    for iobs_time in obs_time:
                        irfs_file_name = f'Prod5-{isite}-{izenith}-{iopti}-{isite_array[1]}.{iobs_time[0]}-v0.1.fits.gz'
                        file_name = f'{dir_fits}{dir_FITS}{irfs_file_name}'
                        ctao_irfs = load_irf_dict_from_file(file_name)
                        ctao_irfs_list.append(ctao_irfs)
                        irfs_label = isite_array[0] + ' (' + izenith + '-' + iobs_time[1] + ')'
                        irfs_label_list.append(irfs_label)
    
        self.irfs_list = ctao_irfs_list
        self.irfs_label_list = irfs_label_list


# In[46]:


class ObservationParameters:
    """Container for observation parameters.

    Parameters
    ----------
    livetime :  `~astropy.units.Quantity`
        Livetime exposure of the simulated observation
    on_region_radius : `~astropy.coordinates.angles.Angle`
        Integration radius of the ON extraction region
    offset : `~astropy.units.Quantity`
        Pointing position offset    
    e_edges_min :  `~astropy.units.Quantity`
        Minimal energy for simulation
    e_edges_max : `~astropy.units.Quantity`
        Maximal energy for simulation
    n_obs : int
    
        Number of simulations of each observation   
    alpha : `~astropy.units.Quantity`
        Normalisation between ON and OFF regions
    """
    @u.quantity_input(livetime=u.h, on_region_radius=u.deg, offset=u.deg, e_edges_min=u.eV, e_edges_max=u.eV)
    def __init__(self,livetime=None,
                 on_region_radius=None, 
                 offset=None, 
                 e_edges_min=None,
                 e_edges_max=None,
                n_obs=None, 
                ):
        self.livetime = livetime
        self.on_region_radius = on_region_radius
        self.offset = offset
        self.e_edges_min = e_edges_min
        self.e_edges_max = e_edges_max
        self.n_obs = n_obs

    def __str__(self):
        """Observation summary report (`str`)."""
        ss = '*** Observation parameters summary ***\n\n'
        ss += 'livetime={} [{}]\n'.format(self.livetime.value, self.livetime.unit)
        ss += 'on_region_radius={} [{}]\n'.format(self.on_region_radius.value, self.on_region_radius.unit)
        ss += 'offset={} [{}]\n'.format(self.offset.value, self.offset.unit)
        ss += 'e_edges_min={} [{}]\n'.format(self.e_edges_min.value, self.e_edges_min.unit)
        ss += 'e_edges_max={} [{}]\n'.format(self.e_edges_max.value, self.e_edges_max.unit)
        ss += 'n_obs={}\n'.format(self.n_obs)
        return ss


# In[42]:





# In[ ]:





# In[47]:


# from astropy.coordinates import SkyCoord, Angle

# obs_parm=ObservationParameters(
#     livetime=50*u.h, 
#     offset=0.11*u.deg, 
#     e_edges_min=0.1*u.TeV, 
#     e_edges_max=100.*u.TeV,
#     on_region_radius=Angle("1.0 deg"),
#     n_obs=1000
# )
# print(obs)


# In[ ]:


# irf_zenith=20
# irf_hours=0.5
# site='North'
# ctaoirfs = CTAOIrfs(irf_zenith, irf_hours, site)


# In[ ]:


# ctao_perf=CTAOPerf()


# In[ ]:


# ctao_perf.load_irfs()


# In[ ]:


# len(ctao_perf.irfs)


# In[ ]:


# cta_perf_list


# In[ ]:




