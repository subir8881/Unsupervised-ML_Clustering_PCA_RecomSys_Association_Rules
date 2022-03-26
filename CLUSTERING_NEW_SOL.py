#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np


# In[2]:


from matplotlib import pyplot as plt
import seaborn as sns


# In[3]:


CRIME = pd.read_csv('crime_data.csv')


# In[4]:


CRIME.head()


# In[5]:


CRIME.shape


# In[6]:


CRIME.isnull().sum()


# In[7]:


CRIME.describe()


# In[8]:


CRIME.value_counts


# In[9]:


CRIME.corr


# In[10]:


CRIME.info()


# In[11]:


CRIME[CRIME.duplicated()].sum()


# In[12]:


CRIME.rename(columns={"Unnamed: 0":"state"},inplace=True)


# In[13]:


fig, ax = plt.subplots(1, 4, figsize=(25,7))
sns.distplot(CRIME.Murder, ax=ax[0])
sns.distplot(CRIME.Assault, ax=ax[1])
sns.distplot(CRIME.UrbanPop, ax=ax[2])
sns.distplot(CRIME.Rape, ax=ax[3])
plt.tight_layout()
plt.show()


# In[14]:


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[15]:


df_norm = norm_func(CRIME.iloc[:,1:])


# In[16]:


from scipy.cluster.hierarchy import linkage 

import scipy.cluster.hierarchy as sch


# In[17]:


type(df_norm)


# In[18]:


df_norm.head()


# In[19]:


help(linkage)
z = linkage(df_norm, method="complete",metric="euclidean")


# In[20]:


plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,leaf_rotation=0., 
    leaf_font_size=8.,
    )
plt.show()


# In[21]:


help(linkage)


# In[22]:


h_complete=AgglomerativeClustering(n_clusters=3,linkage='complete',affinity = "euclidean").fit(df_norm) 


# In[23]:


N_h_complete=h_complete.fit_predict(df_norm)
clusters= pd.DataFrame(N_h_complete,columns=['Clusters'])


# In[24]:


clusters


# In[25]:


dendrogram = sch.dendrogram(sch.linkage(df_norm, method='single'))


# In[26]:


H_C = AgglomerativeClustering(n_clusters=2, affinity = 'euclidean', linkage = 'single')


# In[27]:


y_H_C =H_C.fit_predict(df_norm)
Clusters=pd.DataFrame(y_H_C,columns=['Clusters'])


# In[28]:


CRIME['cluster'] = y_H_C


# In[29]:


CRIME.groupby('cluster').agg(['mean']).reset_index()


# In[30]:


for i in range(2):
    print("cluster", i)
    print("The Members:", ' | '.join(list(CRIME[CRIME['cluster'] == i]['state'].values)))
    print("Total Members:", len(list(CRIME[CRIME['cluster'] == i]['state'].values)))
    print()


# In[31]:


c = pd.read_csv("crime_data.csv")
c.rename(columns={"Unnamed: 0":"state"},inplace=True)


# In[32]:


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
df_norm = norm_func(c.iloc[:,1:])


# In[33]:


dendrogram = sch.dendrogram(sch.linkage(df_norm, method='average'))


# In[34]:


HC = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'average')


# In[35]:


y_h_c = HC.fit_predict(df_norm)
Clusters=pd.DataFrame(y_h_c,columns=['Clusters'])


# In[36]:


c['cluster'] = y_h_c


# In[37]:


c.groupby('cluster').agg(['mean']).reset_index()


# In[38]:


for i in range(4):
    print("cluster", i)
    print("The Members:", ' | '.join(list(c[c['cluster'] == i]['state'].values)))
    print("Total Members:", len(list(c[c['cluster'] == i]['state'].values)))
    print()


# In[39]:


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
df_norm = norm_func(c.iloc[:,1:])


# In[40]:


dendrogram = sch.dendrogram(sch.linkage(df_norm, method='complete'))


# In[41]:


hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'complete')


# In[42]:


y_hc = hc.fit_predict(df_norm)
Clusters=pd.DataFrame(y_hc,columns=['Clusters'])


# In[43]:


c['cluster'] = y_hc


# In[44]:


c.groupby('cluster').agg(['mean']).reset_index()


# In[45]:


for i in range(4):
    print("cluster", i)
    print("The Members:", ' | '.join(list(c[c['cluster'] == i]['state'].values)))
    print("Total Members:", len(list(c[c['cluster'] == i]['state'].values)))
    print()


# In[46]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_crime_df = scaler.fit_transform(c.iloc[:,1:])


# In[47]:


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
df_norm = norm_func(c.iloc[:,1:])


# In[48]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(df_norm)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[49]:


from sklearn.cluster import KMeans
clusters_new = KMeans(5, random_state=42)
clusters_new.fit(df_norm)


# In[50]:


km_label=clusters_new.labels_


# In[51]:


c['cluster'] = clusters_new.labels_


# In[52]:


clusters_new.cluster_centers_


# In[53]:


c.groupby('cluster').agg(['mean']).reset_index()


# In[54]:


for i in range(4):
    print("cluster", i)
    print("The Members:", ' | '.join(list(c[c['cluster'] == i]['state'].values)))
    print("Total Members:", len(list(c[c['cluster'] == i]['state'].values)))
    print()

