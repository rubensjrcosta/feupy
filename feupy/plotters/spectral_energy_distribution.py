#!/usr/bin/env python
# coding: utf-8

# In[1]:


from feupy.config import *


# In[6]:





# In[ ]:


__all__ = [
    "show_SED",
]


# In[2]:


from astropy import units as u
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt # A collection of command style functions

def show_SED(
    datasets = None,  
    models = None,
    dict_leg_style = None, 
    sed_type = "e2dnde", 
    dict_plot_axis =  dict(
        label =  (r'$\rm{E\ [TeV] }$', r'$\rm{E^2\ J(E)\ [TeV\ cm^{-2}\ s^{-1}] }$'),
        units =  (          'TeV',                       'TeV  cm-2     s-1')
    ),
    dict_plot_limits = dict(
        energy_bounds = [1e-5, 3e2] * u.TeV,
        ylim = [1e-23, 1e-7]
    ),
    dict_leg_place = dict(
#         bbox_to_anchor = (0, -0.45), # Set legend outside plot
        ncol=3, 
        loc='lower left', 
    ),
    file_path=None
):    
    
    ax = plt.subplot()
    
    ax.xaxis.set_units(u.Unit(dict_plot_axis['units'][0]))
    ax.yaxis.set_units(u.Unit(dict_plot_axis['units'][1]))

    kwargs = {
        "ax": ax, 
        "sed_type": sed_type,
#         "uplims": True
    }
                        
    for index, dataset in enumerate(datasets):
        color = dict_leg_style[dataset.name][0]
        marker = dict_leg_style[dataset.name][1]
        
        label =    dataset.name
        dataset.data.plot(
            label = label, 
            marker = marker, 
            color=color,
            **kwargs
        )
    
    if models: 
        for index, model in enumerate(models):
            linestyle = dict_leg_style[model.name][1]
            color = dict_leg_style[model.name][0]
            spectral_model = model.spectral_model
            
            energy_bounds=dict_plot_limits['energy_bounds']

            spectral_model.plot(label = f"{model.name} (fit)", energy_bounds=energy_bounds,   marker = ',', color="black", **kwargs)
#             energy_bounds = [7e-2, 8e2] * u.TeV
#             spectral_model.plot(energy_bounds=energy_bounds,  linestyle = linestyle,  marker = ',', color=color, **kwargs)
#             spectral_model.plot(label = f"{model.name}", energy_bounds=energy_bounds,  linestyle = linestyle, color=color, **kwargs)

            spectral_model.plot_error(energy_bounds=energy_bounds,**kwargs)
    
    ax.set_ylim(dict_plot_limits['ylim'])
    ax.set_xlim(dict_plot_limits['energy_bounds'])
    
    ax.legend(**dict_leg_place)
    
    plt.xlabel(dict_plot_axis['label'][0])   
    plt.ylabel(dict_plot_axis['label'][1])
    
    if file_path:
        plt.savefig(file_path, bbox_inches='tight')
#    plt.grid(which="both")
    plt.show()
    
    return


# In[ ]:




