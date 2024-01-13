#!/usr/bin/env python
# coding: utf-8

# In[1]:


from feupy.config import *


# In[6]:


from gammapy.modeling.models import Models
from gammapy.datasets import Datasets


# In[ ]:


__all__ = [
    "set_leg_style_models",
    "set_leg_style_datasets",
    "set_leg_style",
]


# In[2]:


def set_leg_style_models(leg_style, models, color = None, linestyle = None):
    models = Models(models)
    color_m = color
    linestyle_m = linestyle
    
    if not linestyle:
        while len(LINESTYLES) < len(models) +1:
            LINESTYLES.extend(LINESTYLES)
    if not color_m:      
        while len(COLORS) < len(models) +1:
            COLORS.extend(COLORS)

    for index, model in enumerate(models):
        if not color_m:
            color = "black"
            
        linestyle = LINESTYLES[index]
        leg_style[model.name] = (color, linestyle)
    return leg_style


# In[ ]:


def set_leg_style_datasets(leg_style, datasets, color = None, marker = None):
    datasets = Datasets(datasets)
    marker_ds = marker
    color_ds = color
    if not marker_ds:
        while len(MARKERS) < len(datasets) +1:
            MARKERS.extend(MARKERS)
    if not color_ds:      
        while len(COLORS) < len(datasets) +1:
            COLORS.extend(COLORS)

    for index, dataset in enumerate(datasets):
        if not color_ds:
            color = COLORS[index]

        if not color_ds:
            marker = MARKERS[index]
        
        #############################
        if dataset.name.find('LHAASO') != -1:
            color = COLOR_LHAASO
            marker = MARKER_LHAASO
            
        if dataset.name.find('CTA') != -1:
            color = COLOR_CTA
            marker = MARKER_CTA
        #############################    
        leg_style[dataset.name] = (color, marker)
    return leg_style


# In[3]:


def set_leg_style(leg_style, datasets = None, models = None, color = None, marker = None, linestyle = None):
    if all([datasets ==  None, models ==  None]):
        return print("Sorry, there is error: 'datasets =  None' and 'models =  None'")
    else: 
        if datasets !=  None:
            leg_style = set_leg_style_datasets(leg_style, datasets, color, marker)

        if models !=  None:
            leg_style = set_leg_style_models(leg_style, models, color, linestyle)
        
    return leg_style


# In[ ]:




