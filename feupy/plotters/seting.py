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


def set_leg_style_models(dict_leg_style, models, color = None, linestyle = None):
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
        dict_leg_style[model.name] = (color, linestyle)
    return dict_leg_style


# In[ ]:


def set_leg_style_datasets(dict_leg_style, datasets, color = None, marker = None):
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
        dict_leg_style[dataset.name] = (color, marker)
    return dict_leg_style


# In[3]:


def set_leg_style(dict_leg_style, datasets = None, models = None, color = None, marker = None, linestyle = None):
    if all([datasets ==  None, models ==  None]):
        return print("Sorry, there is error: 'datasets =  None' and 'models =  None'")
    else: 
        if datasets !=  None:
            dict_leg_style = set_leg_style_datasets(dict_leg_style, datasets, color, marker)

        if models !=  None:
            dict_leg_style = set_leg_style_models(dict_leg_style, models, color, linestyle)
        
    return dict_leg_style


# In[4]:


# class ROI:
#     # ADD others parameters
#     all=[]

#     # Validating the units of arguments to functions
#     @u.quantity_input(pos_ra=u.deg, pos_dec=u.deg, radius=u.deg)
#     def __init__(self, name: str, pos_ra, pos_dec, radius):

#         # Run validations to the received arguments
#         assert 0 <= pos_ra.value <= 360, f"Right Ascension {pos_ra} is not in the range: (0,360) deg!"
#         assert -90 <= pos_dec.value <= 90, f"Declination {pos_dec} is not in the range: (-90,90) deg!"

#         # Assign to self object
#         self.__name=name
#         self.radius=radius
#         self.position=SkyCoord(pos_ra,pos_dec) # convert coordinates to astropy SkyCoord

#         # Actions to execute
#         ROI.all.append(self) 

#     @property
#     def info(self):
#         info={}
#         info["name"]=self.__name
#         info["position"]=self.position
#         info["radius"]=self.radius
#         return info

#     @property
#     # Property Decorator=Read-Only Attribute
#     def name(self):
#         return self.__name

#     def __repr__(self):
#         return f"{self.__class__.__name__}({self.__name}, {(self.position.ra.value)}.deg, {(self.position.dec.value)}.deg, {(self.radius.value)}.deg)"


# In[5]:


# def test_roi():
#     return ROI(
#         "LHAASO J1825-1326", 
#         u.Quantity("276.45deg"), 
#         -13.45* u.Unit('deg'), 
#         u.Quantity("1.0deg")
#     )


# In[ ]:




