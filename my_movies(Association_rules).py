#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install mlxtend')


# In[2]:


import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt


# In[3]:


movie=pd.read_csv('my_movies.csv')
movie.head()


# In[4]:


movie=movie.drop(['V1','V2','V3','V4','V5'], axis=1)
movie


# In[5]:


movie.shape


# #Apriori Algorithms

# ####If support value is 0.2 and threshold is 0.6.
# 

# In[6]:


frequent_itemsets = apriori(movie, min_support=0.2, use_colnames=True)
frequent_itemsets


# In[7]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.6)
rules
rules.sort_values('lift',ascending = False)


# In[8]:


rules[rules.lift>1]
#(Green Mile)	(Sixth Sense)	,
#(Patriot)	(Gladiator) ,
#(LOTR2)	(LOTR1) ,
#(LOTR1)	(LOTR2) ,
#(Patriot, Sixth Sense)	(Gladiator)   are all at high cofidence at 100%.


# In[9]:


rules.sort_values('lift',ascending = False)


# In[10]:


plt.scatter('support','confidence',data=rules)


# ###If support value is 0.5 and threshold is 1.

# In[11]:


frequent_itemsets = apriori(movie, min_support=0.5, use_colnames=True)
frequent_itemsets


# In[12]:


rules1 = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules1
rules1.sort_values('lift',ascending = False)


# In[13]:


rules1.sort_values('lift',ascending = False)


# In[14]:


rules1[rules.lift>1]
#(Patriot)	(Gladiator) are high confidence at 100%.


# In[15]:


plt.scatter('support','confidence',data=rules1)


# #Inference

# If support vule is 0.2 and threshold is 0.6 then(Green Mile)	(Sixth Sense)	,
# (Patriot)	(Gladiator) ,(LOTR2)	(LOTR1) ,(LOTR1)	(LOTR2) ,(Patriot, Sixth Sense)	(Gladiator)   are all at 100% confidence.
# 
# 
# If support value is 0.5 and threshold is 1 then (Patriot)	(Gladiator) are at the 100% confidence.

# In[ ]:




