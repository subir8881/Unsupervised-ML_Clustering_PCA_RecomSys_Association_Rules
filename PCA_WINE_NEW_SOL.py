#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale


# In[2]:


WINE = pd.read_csv("wine.csv")


# In[3]:


WINE.describe()


# In[4]:


WINE.head()


# In[5]:


WINE.info()


# In[6]:


WINE.isnull().sum()


# In[7]:


WINE.shape


# In[8]:


import seaborn as sns


# In[9]:


fig, ax = plt.subplots(6, 3, figsize=(15,7))
sns.distplot(WINE.Alcohol,ax=ax[0,0])
sns.distplot(WINE.Malic,ax=ax[0,1])
sns.distplot(WINE.Ash,ax=ax[0,2])
sns.distplot(WINE.Alcalinity,ax=ax[1,0])
sns.distplot(WINE.Magnesium,ax=ax[1,1])
sns.distplot(WINE.Phenols,ax=ax[1,2])
sns.distplot(WINE.Flavanoids,ax=ax[2,0])
sns.distplot(WINE.Nonflavanoids,ax=ax[2,1])
sns.distplot(WINE.Proanthocyanins,ax=ax[2,2])
sns.distplot(WINE.Color,ax=ax[3,0])
sns.distplot(WINE.Hue,ax=ax[3,1])
sns.distplot(WINE.Dilution,ax=ax[3,2])
sns.distplot(WINE.Proline,ax=ax[4,0])
plt.tight_layout()
plt.show()


# In[10]:


WINE.Data = WINE.iloc[:,1:]
WINE.Data.head()


# In[11]:


W = WINE.Data.values


# In[12]:


W_normal = scale(W)


# In[13]:


W_normal


# In[14]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


# In[15]:


fig=plt.figure(figsize=(15,12))
dendrogram = sch.dendrogram(sch.linkage(W_normal, method='average'))
plt.title("Dendrogram",size=15)


# In[16]:


pca = PCA()
pca_values = pca.fit_transform(W_normal)
pca_values


# In[17]:


pca = PCA(n_components=13)
pca_values = pca.fit_transform(W_normal)


# In[18]:


var = pca.explained_variance_ratio_
var


# In[19]:


var1 = np.cumsum(np.round(var,decimals= 4)*100)
var1


# In[20]:


pca.components_


# In[21]:


plt.plot(var1,color="red")


# In[22]:


pca_values[:,0:13]


# In[23]:


x = pca_values[:,0:1]
y = pca_values[:,1:2]
plt.scatter(x,y)


# In[24]:


pca.components_


# In[25]:


pca_values.shape


# In[26]:


finalDf= pd.concat([pd.DataFrame(pca_values[:,0:2],columns=['pc1', 'pc2']), WINE[['Type']]], axis= 1)


# In[27]:


import seaborn as sns
sns.scatterplot(data=finalDf,x='pc1', y='pc2', hue='Type')


# In[28]:


new_df = pd.DataFrame(pca_values[:,0:13])


# In[29]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3)
kmeans.fit(new_df)
kmeans.labels_


# In[30]:


#clustering with the first 3 PCA
wine_pca=pca_values[:,0:3]


# In[31]:


fig=plt.figure(figsize=(15,12))
dendrogram = sch.dendrogram(sch.linkage(wine_pca, method='ward'))
plt.title("Dendrogram",size=15)


# In[32]:


hc1_principal = AgglomerativeClustering(n_clusters=3, affinity = 'euclidean', linkage = 'ward')


# In[33]:


y_hc1_principal = hc1_principal.fit_predict(wine_pca)
Clusters=pd.DataFrame(y_hc1_principal,columns=['Clusters'])


# In[34]:


WINE['Cluster'] = y_hc1_principal


# In[35]:


WINE.groupby('Cluster').agg(['mean']).reset_index()


# In[36]:


for i in range(3):
    print("Cluster", i)
    print("Total Members:", len(list(WINE[WINE['Cluster'] == i]['Type'].values)))
    print()


# In[37]:


fig=plt.figure(figsize=(15,12))
dendrogram = sch.dendrogram(sch.linkage(wine_pca, method='average'))
plt.title("Dendrogram",size=15)


# In[38]:


hc2_principal = AgglomerativeClustering(n_clusters=5, affinity = 'euclidean', linkage = 'average')


# In[39]:


y_hc2p = hc2_principal.fit_predict(wine_pca)
Clusters=pd.DataFrame(y_hc2p,columns=['Clusters'])


# In[40]:


WINE['cluster'] = y_hc2p


# In[41]:


WINE.groupby('cluster').agg(['mean']).reset_index()


# In[42]:


for i in range(5):
    print("Cluster", i)
    print("Total Members:", len(list(WINE[WINE['Cluster'] == i]['Type'].values)))
    print()


# In[ ]:





# In[ ]:





# In[ ]:




