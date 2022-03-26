#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pylab as plt 
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import numpy as np


# In[3]:


AIR= pd.read_excel('EastWestAirlines2.xlsx')


# In[4]:


AIR


# In[19]:


def norm_func(i):
    x=(i-i.min()/i.std())
    return (x)
df_norm=norm_func(AIR.iloc[:,1:])


# In[20]:


df_norm


# In[21]:


from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 
help(linkage)


# In[22]:


z=linkage(df_norm,method="complete",metric="euclidean")


# In[23]:


plt.figure(figsize=(25,50));plt.title("Dendogram");plt.xlabel("Index");plt.ylabel("Distance")
sch.dendrogram(
    z,
    leaf_rotation=0.,
    leaf_font_size=6.,
)
plt.show()


# In[24]:


h_labels=AgglomerativeClustering(n_clusters=10,affinity="euclidean",linkage="complete").fit(df_norm)
clusters_labels=pd.Series(h_labels.labels_)


# In[27]:


AIR["Clusters"]=clusters_labels
df_final=AIR.iloc[:,[0,12,1,2,3,4,5,6,7,8,9,10,11]]


# In[28]:


df_final


# In[29]:


df_final.to_csv("Flights.csv")

