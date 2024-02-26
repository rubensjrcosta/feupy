#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gammapy.data import Observation


# In[2]:


__all__ = [
    "create_observation",
]


# In[ ]:


def create_observation(pointing, livetime, irfs, location):
    """Create an observation."""
    return Observation.create(
        pointing=pointing,
        livetime=livetime,
        irfs=irfs,
        location=location,
    )


# In[ ]:




