#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as plt


# In[2]:


book=pd.read_csv('book.csv',encoding='latin-1')
book


# In[3]:


df =book.drop(['Unnamed: 0'],axis=1)


# In[4]:


df =df.rename({'User.ID':'user_id','Book.Title':'book_title','Book.Rating':'book_rating'},axis=1)


# In[5]:


df.info()


# In[6]:


len(df.user_id.unique())


# In[7]:


len(df.book_title.unique())


# In[8]:


df1 = df.drop_duplicates(['user_id','book_title'])


# In[9]:


books = df1.pivot(index='user_id',
                                 columns='book_title',
                                 values='book_rating').reset_index(drop=True)


# In[10]:


books


# In[11]:


books.index = df.user_id.unique()


# In[12]:


books


# In[13]:


books.fillna(0, inplace=True)


# In[14]:


books


# In[15]:


from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation


# In[16]:


df2 = 1 - pairwise_distances( books.values,metric='cosine')


# In[17]:


df2


# In[18]:


#Store the results in a dataframe
books2 = pd.DataFrame(df2)


# In[19]:


books2.index = df1.user_id.unique()
books2.columns = df1.user_id.unique()


# In[20]:


books2.iloc[0:5, 0:5]


# In[21]:


np.fill_diagonal(df2, 0)
books2.iloc[0:5, 0:5]


# In[22]:


books2.idxmax(axis=1)[0:5]

