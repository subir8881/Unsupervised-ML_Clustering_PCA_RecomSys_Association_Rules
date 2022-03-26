#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from sklearn.cluster import KMeans


# In[2]:


AIR= pd.read_excel('EastWestAirlines2.xlsx')


# In[3]:


AIR.head()


# In[4]:


def norm_func(i):
    x=(i-i.min()/i.std())
    return (x)
df_norm=norm_func(AIR.iloc[:,1:]) 


# In[5]:


df_norm.head()


# In[6]:


k = list(range(2,15))
k


# In[7]:


wcss = []
for i in range(2,15):
    kmeans= KMeans(n_clusters=i, random_state=0)
    kmeans.fit(df_norm)
    wcss.append(kmeans.inertia_)


# In[8]:


plt.plot(range(2,15),wcss)
plt.title('Elbow Curve')
plt.xlabel('numer of clusters')
plt.ylabel('wcss')
plt.show


# In[9]:


model=KMeans(n_clusters=7)
model.fit(df_norm)


# In[10]:


model.labels_


# In[11]:


ND= pd.Series(model.labels_)
AIR['clust'] = ND
AIR


# In[12]:


AIR.iloc[:,1:12].groupby(AIR.clust).mean()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




