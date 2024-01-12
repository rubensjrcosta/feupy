#!/usr/bin/env python
# coding: utf-8

# In[2]:





# In[10]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst
from psrqpy import QueryATNF
def query_ATNF():
    _table = QueryATNF
    return _table 

class SourceCatalogATNF():
    """ATNF Pulsar Catalogue.

    See: https://www.atnf.csiro.au/research/pulsar/psrcat/

    One source is represented by `~feupy.catalog.SourceCatalogATNF`.
    """    
    tag = "atnf"
    description = "An online catalog of pulsars"
    
    """Pulsar default parameters"""
    PSR_PARAMS =['JNAME', 'RAJD', 'DECJD','RAJ', 'DECJ','DIST','DIST_DM', 'AGE', 'P0','BSURF','EDOT', 'TYPE', 'Assoc']

    def __init__(self):
        self.__query = query_ATNF()

    def table(self):
        return self.__query().table

    def pandas(self):
        return self.__query().pandas

    @property
    # Property Decorator = Read-Only Attribute
    def query(self):
        return self.__query
    
    @property        
    def version(self):
        return __query.get_version


# In[ ]:


from astropy.coordinates import SkyCoord


# In[21]:


from astropy import units as u

class SourceCatalogObjectATNF():
    all = []
    
    # Validating the units of arguments to functions
    @u.quantity_input(
        pos_ra=u.deg, 
        pos_dec=u.deg,
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
        

        # Run validations to the received arguments
        # Run validations to the received arguments
        assert  0 <= pos_ra.value <= 360, f"Right Ascension {pos_ra} is not in the range: (0,360) deg!"
        assert -90 <= pos_dec.value <= 90, f"Declination {pos_dec} is not in the range: (-90,90) deg!"
    
        # Assign to self object
        self.__name = name
        self.position = SkyCoord(pos_ra,pos_dec)
        
        self.dist = dist
        self.age = age
        self.B_surf = B_surf
        self.dist = dist
        self.P_0 = P_0
        self.E_dot = E_dot
        self.assoc = assoc

        SourceCatalogObjectATNF.all.append(self) 
    @property
    # Property Decorator = Read-Only Attribute
    def name(self):
        return self.__name
    
    
    @classmethod
    def instantiate_from_ATNF(cls, _dict):
        for index, data in enumerate(_dict):        
            name = f"PSR {data['JNAME']}" 
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

            SourceCatalogObjectATNF(name=name,pos_ra=pos_ra,pos_dec=pos_dec, dist=dist)

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.__name}', {self.position.ra.deg}.deg, {self.position.dec.deg}.deg)"
    


# In[23]:


# from astropy.coordinates import SkyCoord

# source1 = SourceCatalogObjectATNF(
#     "2HWC J1825-134", 
#     27.46*u.Unit('deg'), 
#     12.2* u.Unit('deg'),
# #     spectral_model=catalogs_roi[1][0].spectral_model(),
# #     flux_points=catalogs_roi[1][0].flux_points,
# #     catalog=catalogs_roi[1].tag
        
# )


# In[ ]:




