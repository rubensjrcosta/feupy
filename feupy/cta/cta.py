#!/usr/bin/env python
# coding: utf-8

# In[22]:


from gammapy.irf import load_irf_dict_from_file


# In[ ]:


__all__ = [
    "CTA",
]


# In[25]:


class CTA:
    """CTA.

    CTA is represented by `~feupy.cta.CTA`.
    """    
#     IRF_ZENITH = [20, 40, 60]
#     IRF_HOURS = [0.5, 5, 50]
    SITE = [("cta_north", "North"), ("cta_south", "South")]

    def __init__(self):
        self.irf_name = None
    
    def create_irf_name(self, irf_zenith=20, irf_hours=0.5, site='North'):
        return f'{site}_z{irf_zenith}_{irf_hours}h'
    
    def load_irfs(self, irf_zenith=20, irf_hours=0.5, site="North"):
        dirbasename="$PYTHONPATH/feupy/data/irfs/CTAO_IRFs-prod5versionv0.1/fits/"
        
        if site == 'South':
            telescopes = '14MSTs37SSTs'
        else: telescopes = '4LSTs09MSTs'
        
        dir_irf = f'CTA-Performance-prod5-v0.1-{site}-{irf_zenith}deg.FITS/'
        
        if irf_hours == 0.5:
            seconds = 1800
        elif irf_hours == 5:
            seconds = 18000
        else: seconds = 180000
            
        irf_file_name = f'Prod5-{site}-{irf_zenith}deg-{site}Az-{telescopes}.{seconds}s-v0.1.fits.gz'
        return load_irf_dict_from_file(f'{dirbasename}{dir_irf}{irf_file_name}')


# In[26]:


# CTA().load_irfs()


# In[ ]:





# In[ ]:




