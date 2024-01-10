#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from .source import Source

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from gammapy.utils.scripts import make_path


# In[ ]:


__all__ = [
    "Pulsar",
]


# In[2]:


class Pulsar(Source):
    all = []
    
    # Validating the units of arguments to functions
    @u.quantity_input(
        age= u.yr, 
        B_surf= u.G, 
        P_0 = u.s, 
        E_dot = u.Unit('erg -1'), 
        dist=u.pc
    )
    def __init__(self, 
                 name: str, 
                 pos_ra,  
                 pos_dec, 
                 age = None, 
                 B_surf= None,
                 P_0= None, 
                 E_dot= None, 
                 assoc= None, 
                 dist=None
                ):
# JName: Pulsar name based on J2000 coordinates
# Right ascension (J2000) (degrees)
# Declination (J2000) (degrees
# Age: Spin down age (yr) []
# Dist: Best estimate of the pulsar distance using the YMW16 DM-based distance as default (kpc)

        # Call to super function to have access to all attributes / methods
        super().__init__(name, 
                         pos_ra, 
                         pos_dec
        )
        

        # Run validations to the received arguments
        assert  dist.value>=0, f"Distance: {dist} <= 0!"
    
        # Assign to self object
        self.dist = dist
        self.age = age
        self.B_surf = B_surf
        self.dist = dist
        self.P_0 = P_0
        self.E_dot = E_dot
        self.assoc = assoc
        self.pos = SkyCoord(pos_ra,pos_dec)

        Pulsar.all.append(self) 

    @classmethod
    def instantiate_from_fits(cls):
        file_name = "$PYTHONPATH/feupy/data/catalogs/pulsars-catalog.fits"
        table = Table().read(make_path(file_name))
        col_names = table.colnames
        for index, pulsar in enumerate(table):
            index, name, ra, dec, distance, age, luminosity = table[index][col_names]
            pos_ra = ra*table["ra"].unit
            pos_dec = dec*table["dec"].unit
            dist = distance*table["distance"].unit
            age = age*table["age"].unit
            E_dot = luminosity*table["luminosity"].unit
            Pulsar(name=name,pos_ra=pos_ra,pos_dec=pos_dec,dist=dist)
    
    @classmethod
    def instantiate_from_ATNF(cls, _dict):
        for index, data in enumerate(_dict):        
            name = data['JNAME'] 
            pos_ra =  data['RAJD'] 
            pos_dec = data['DECJD'] 
            age = data['AGE'] 
            dist = data['DIST']

            # self.B_surf = data['BSURF'] # BSurf: Surface magnetic flux density (Gauss) []
            # self.P_0 = data['P0'] # P0: Barycentric period of the pulsar (s)
            # self.E_dot = data['EDOT'] # Edot: Spin down energy loss rate (ergs/s))
            # self.assoc = data['ASSOC'] # Assoc: Names of other objects, e.g., supernova remnant, globular cluster or gamma-ray source associated with the pulsar
            # self.rajd =  Quantity(data['RAJ'], u.hourangle)
            # self.decjd = Quantity(data['DECJ'], u.deg)

            Pulsar(name=name,pos_ra=pos_ra,pos_dec=pos_dec, dist=dist)


# In[ ]:




