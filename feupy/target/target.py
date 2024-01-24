#!/usr/bin/env python
# coding: utf-8

# In[1]:


from astropy import units as u
from astropy.coordinates import SkyCoord

from gammapy.modeling.models import SkyModel, SpectralModel, SpatialModel, TemporalModel


# In[2]:


__all__ = [
    "Target",
]


# In[131]:


class Target:
    """Observation target information.
    
    Parameters
    ----------
    name : `str`
        Name of the source
    pos_ra : `~astropy.units.Quantity`
        Right ascension (J2000) (degrees) of the source position
    pos_dec : `~astropy.units.Quantity`
        Declination (J2000) (degrees) of the source position
    spectral_model : `~gammapy.modeling.models.SpectralModel`
        Spectral Model of the source
    spatial_model : `~gammapy.modeling.models.SpatialModel`
        Spatial Model of the source
    temporal_model : `~gammapy.modeling.models.TemporalModel`
        Temporal Model of the source
    """
    
    all = []
    # Validating the units of arguments to functions
    @u.quantity_input(pos_ra=u.deg, pos_dec=u.deg)
    def __init__(
        self, 
        name: str, 
        pos_ra, 
        pos_dec,
        spectral_model: SpectralModel=None,
        spatial_model: SpatialModel=None,
        temporal_model: TemporalModel=None,
    ):

        # Run validations to the received arguments
        assert  0 <= pos_ra.value <= 360, f"Right Ascension {pos_ra} is not in the range: (0,360) deg!"
        assert -90 <= pos_dec.value <= 90, f"Declination {pos_dec} is not in the range: (-90,90) deg!"
        
        # Assign to self object
        self.__name = name
        self.position = SkyCoord(pos_ra, pos_dec) # convert coordinates to astropy SkyCoord
        self.spectral_model = spectral_model
        self.spatial_model = spatial_model
        self.temporal_model = temporal_model
        
        Target.all.append(self) 
        
    @property
    def name(self):
        return self.__name
        
    @property
    def info(self):
        """Target report (`str`)."""
        ss = '*** Target parameters ***\n\n'
        ss += 'Name={}\n'.format(self.name)
        ss += "pos_ra={:.2f}\n".format(self.position.ra).replace(' ', '')
        ss += "pos_dec={:.2f}\n".format(self.position.dec).replace(' ', '')
        if self.spectral_model:
            ss += '\n*** Spectral Model parameters ***\n\n'
            for par in self.spectral_model.parameters:
                ss += '{}={} {}\n'.format(par.name, str(par.value), par.unit)
        if self.spatial_model:
            ss += '\n*** Spatial Model parameters ***\n\n'
            for par in self.spatial_model.parameters:
                ss += '{}={} {}\n'.format(par.name, str(par.value), par.unit)
        if self.temporal_model:
            ss += '\n*** Temporal Model parameters ***\n\n'
            for par in self.temporal_model.parameters:
                ss += '{}={} {}\n'.format(par.name, str(par.value), par.unit)
        return ss
    
    def __repr__(self):
        ss = f"{self.__class__.__name__}("
        ss += f"name={self.name!r}, "
        ss += "pos_ra=Quantity('{:.2f}'), ".format(self.position.ra).replace(' ', '')
        ss += "pos_dec=Quantity('{:.2f}'))\n".format(self.position.dec).replace(' ', '')
        return ss 


# In[132]:


def test_target():
    return Target(
        name="23HWC J1825-134", 
        pos_ra=27.46* u.Unit('deg'), 
        pos_dec=12.2* u.Unit('deg'),
    
    )


# In[79]:


# target = test_target()
# target.spectral_model = LogParabolaSpectralModel()
# target.spatial_model = GaussianSpatialModel()
# target.temporal_model = GaussianTemporalModel()

# print(target.spectral_model.parameters.value)
# print(target.spatial_model.parameters.value)
# print(target.temporal_model.parameters.value)

# for par in target.temporal_model.parameters:
#     print(par.name)

# print(target.all)

# print(target.info)

