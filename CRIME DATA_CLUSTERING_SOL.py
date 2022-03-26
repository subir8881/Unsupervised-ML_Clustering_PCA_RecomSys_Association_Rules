#!/usr/bin/env python
# coding: utf-8

# In[2]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np


# In[3]:


from matplotlib import pyplot as plt
import seaborn as sns


# In[4]:


CRIME = pd.read_csv('crime_data.csv')


# In[5]:


CRIME


# In[6]:


def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)


# In[8]:


df_norm = norm_func(CRIME.iloc[:,1:])


# In[9]:


from scipy.cluster.hierarchy import linkage 

import scipy.cluster.hierarchy as sch


# In[10]:


type(df_norm)


# In[12]:


help(linkage)
z = linkage(df_norm, method="complete",metric="euclidean")


# In[14]:


plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,leaf_rotation=0., 
    leaf_font_size=8.,
    )
plt.show()


# In[15]:


help(linkage)


# In[16]:


h_complete=AgglomerativeClustering(n_clusters=3,linkage='complete',affinity = "euclidean").fit(df_norm) 


# In[28]:


N_h_complete=h_complete.fit_predict(df_norm)
clusters= pd.DataFrame(N_h_complete,columns=['Clusters'])


# In[32]:


CRIME


# In[33]:


clusters

