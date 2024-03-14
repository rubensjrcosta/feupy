#!/usr/bin/env python
# coding: utf-8

# In[1]:


from feupy.plotters.config import *


# In[2]:


from gammapy.modeling.models import Models
from gammapy.datasets import Datasets


# In[3]:


__all__ = [
    "set_leg_style_models",
    "set_leg_style_datasets",
    "set_leg_style",
    "setting_leg_style"
]


# In[4]:


def set_leg_style(
    leg_style = {}, 
    datasets_names=None, 
    models_names=None, 
    colors=COLORS, 
    markers=MARKERS, 
    linestyles=LINESTYLES
):
    
    if all([datasets_names ==  None, models_names ==  None]):
        return print("Sorry, there is error: 'datasets_names =  None' and 'models_names =  None'")
    else: 
        if datasets_names !=  None:
            leg_style = set_leg_style_datasets(datasets_names, leg_style, colors, markers)

        if models_names !=  None:
            leg_style = set_leg_style_models(models_names, leg_style, colors, linestyles)
    
    return leg_style


# In[5]:


def set_leg_style_datasets(datasets_names, leg_style={}, colors=COLORS, markers=MARKERS):
    if not isinstance(datasets_names, list):
        datasets_names = [datasets_names]
        
    if len(colors) < len(datasets_names):      
        while len(colors) < len(datasets_names) +1:
            colors.extend(colors)
    
    if len(markers) < len(datasets_names):
        while len(markers) < len(datasets_names)+1:
            markers.extend(markers)

    for index, name in enumerate(datasets_names):
        color = colors[index]
        marker = markers[index]   
        leg_style[name] = (color, marker)
    return leg_style


# In[6]:


def set_leg_style_models(models_names, leg_style={}, colors='black', linestyles=LINESTYLES):
    if not isinstance(models_names, list):
        models_names = [models_names]
    
    if len(linestyles) < len(models_names):
        while len(linestyles) < len(models_names)+1:
            linestyles.extend(linestyles)

    if len(colors) < len(models_names):      
        while len(colors) < len(models_names) +1:
            colors.extend(colors)

    for index, name in enumerate(models_names):
        color = colors[index]
        linestyle = linestyles[index]   
        leg_style[name] = (color, linestyle)
    return leg_style


# In[7]:


def setting_leg_style( 
    leg_style = {},
    names=None, 
    colors=COLORS, 
    markers=MARKERS, 
    linestyles=LINESTYLES
    
):
    if all([names ==  None]):
        return print("Sorry, there is error: 'names =  None'")
    else: 
        if not isinstance(names, list):
            names = [names]    
    
    if len(colors) < len(names):      
        while len(colors) < len(names) + 1:
            colors.extend(colors)

    if len(markers) < len(names):
        while len(markers) < len(names) + 1:
            markers.extend(markers)
            
    if len(linestyles) < len(names):
        while len(linestyles) < len(names) + 1:
            linestyles.extend(linestyles)
    
    for index, name in enumerate(names):
#     for index, name in enumerate(names):
        leg_style[name] = (colors[index], linestyles[index], markers[index]) 
    return leg_style


# In[ ]:





