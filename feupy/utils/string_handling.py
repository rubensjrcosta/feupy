#!/usr/bin/env python
# coding: utf-8

# In[1]:


def name_to_txt(name):
    return name.replace(" ", "_").replace(":", "")


# In[2]:


# Data column for use in a Table object to string using join()
def column_to_string(column):
    return f'[{",".join(str(element) for element in list(column))}]'


# In[ ]:


# string representation of list to list using json.loads()
import json
def string_to_list(string):
    return json.loads(string)

