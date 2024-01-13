#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json


# In[1]:


def name_to_txt(name):
    return name.replace(" ", "_").replace(":", "")


# In[2]:


# string representation of list to list using json.loads()
def string_to_list(string):
    return json.loads(string)


# In[ ]:




