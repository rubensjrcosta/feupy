#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from gammapy.utils.scripts import make_path
from gammapy.modeling.models import SkyModel
from gammapy.modeling.models import SpectralModel
from gammapy.estimators import FluxPoints
import csv


# In[ ]:


__all__ = [
    "Source",
]


# In[1]:


# How to create a class:
class Source:
    # ADD others parameters
    color = "red" # The color of the flux ponts
    all = []
    # Validating the units of arguments to functions
    @u.quantity_input(pos_ra=u.deg, pos_dec=u.deg)
    def __init__(self, 
                 name: str, 
                 pos_ra, 
                 pos_dec, 
                 catalog=None, 
                 spectral_model: SpectralModel=None, 
                 flux_points: FluxPoints=None,
                 **kwargs):
        # Run validations to the received arguments
        assert  0 <= pos_ra.value <= 360, f"Right Ascension {pos_ra} is not in the range: (0,360) deg!"
        assert -90 <= pos_dec.value <= 90, f"Declination {pos_dec} is not in the range: (-90,90) deg!"
        
        # Assign to self object
        self.__name = name
        self.position = SkyCoord(pos_ra,pos_dec)
        self.catalog = catalog
        if spectral_model:
#             self.spectral_model=spectral_model
            if "spatial_model" in kwargs:
                spatial_model = kwargs["spatial_model"]
            else: spatial_model=None
            if "temporal_model" in kwargs:
                temporal_model = kwargs["temporal_model"]
            else: temporal_model=None
            if "model_name" in kwargs:
                name = kwargs["model_name"]
            else: name=None
            if "datasets_names" in kwargs:
                datasets_names = kwargs["datasets_names"]
            else: datasets_names=None
                
            self.sky_model=SkyModel(
                spectral_model=spectral_model,
                spatial_model=spatial_model,
                temporal_model=temporal_model,
                name=name,
                datasets_names=datasets_names,
            )
            
        if flux_points:
            self.flux_points = flux_points
        
        # Actions to execute
        Source.all.append(self)
        
    @property
    # Property Decorator = Read-Only Attribute
    def name(self):
        return self.__name
    
    @classmethod
    def instantiate_from_csv(cls):
        file_name = "$PYTHONPATH/feupy/sources.csv"
        with open(make_path(file_name), 'r') as f:
            reader = csv.DictReader(f)
            sources = list(reader)

        for source in sources:
            name=source.get('name')
            pos_ra=source.get('pos_ra')
            pos_dec=source.get('pos_dec')

            Source(
                name,
                u.Quantity(pos_ra), 
                u.Quantity(pos_dec)
            )

    @staticmethod
    def is_integer(num):
        # We will count out the floats that are point zero
        # For i.e: 5.0, 10.0
        if isinstance(num, float):
            # Count out the floats that are point zero
            return num.is_integer()
        elif isinstance(num, int):
            return True
        else:
            return False
        
    def __repr__(self):
        return f"{self.__class__.__name__}('{self.__name}', {self.position.ra.deg}.deg, {self.position.dec.deg}.deg, {self.catalog})"


# In[4]:


from gammapy.modeling.models import PowerLawSpectralModel

def test_source():
    return Source(
        "2HWC J1825-134", 
        27.46*u.Unit('deg'), 
        12.2*u.Unit('deg'), 
        catalog="2hwc",
        spectral_model=PowerLawSpectralModel(), 
#         flux_points=fp,
        model_name="name"
    )


# In[ ]:




