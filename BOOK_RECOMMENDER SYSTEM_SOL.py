#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


Book_df=pd.read_csv('BOOK4.csv', sep='\t')


# In[3]:


print (Book_df)


# In[4]:


len(Book_df.UID.unique())


# In[5]:


Book3=Book_df.rename({'BOOK TITLE': 'booktitle'}, axis= 1)


# In[6]:


print (Book3)


# In[7]:


len(Book3.booktitle.unique())


# In[8]:


Book4=Book3.rename({'BOOK RATING': 'bookrating'}, axis= 1)


# In[9]:


print (Book4)


# In[10]:


Book5=Book4.drop('SNO', axis=1)


# In[11]:


print 
(Book5)


# In[12]:


len(Book5.UID.unique())


# In[13]:


len(Book5.bookrating.unique())


# In[16]:


Book5_df= Book5.reset_index(drop=True)


# In[17]:


Book5_df


# In[20]:


Book6= Book5_df.reset_index().pivot_table(values='bookrating', index=['UID'], columns='booktitle')


# In[21]:


Book6


# In[34]:


Book6.index= Book5_df.UID.unique()


# In[24]:


Book6


# In[25]:


Book6.fillna(0, inplace=True)


# In[26]:


Book6


# In[27]:


from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation


# In[28]:


NBOOK= 1 - pairwise_distances(Book6.values,metric='cosine')


# In[29]:


NBOOK


# In[30]:


NBOOKB_df = pd.DataFrame(NBOOK)


# In[32]:


NBOOKB_df


# In[35]:


NBOOKB_df.index= Book5.UID.unique()
NBOOKB_df.columns= Book5.UID.unique()


# In[37]:


NBOOKB_df.shape


# In[38]:


NBOOKB_df.iloc[0:10, 0:10]


# In[39]:


NBOOKB_df.idxmax(axis=1)[0:5]


# In[46]:


Book_df[(Book_df['UID']==276729)| (Book_df['UID']==276726)]


# In[55]:


user_1 = Book5[Book5['UID']==276729]
user_2 = Book5[Book5['UID']==276729]


# In[58]:


user_2.booktitle
user_1.booktitle


# In[60]:


pd.merge(user_1,user_2, on='booktitle', how='outer')


# In[65]:


user_rating = Book6["Twilight"]  


# In[66]:


user_rating


# In[68]:


corr_book = Book6.corrwith(user_rating)


# In[69]:


corr_book2 = pd.DataFrame(corr_book, columns=['Correlation'])
corr_book2.dropna(inplace=True)
corr_book2.head()


# In[71]:


corr_book2[corr_book2['Correlation'] > 0].sort_values(by='Correlation', ascending=False).head(10)  


# In[ ]:




