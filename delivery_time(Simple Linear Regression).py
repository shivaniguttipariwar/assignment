#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf


# In[2]:


dt=pd.read_csv('delivery_time.csv')
dt


# In[3]:


dt.describe()


# ###Correlation

# In[4]:


dt.corr()


# In[5]:


dt.hist()


# In[6]:


dt=dt.rename({'Delivery Time':'delivery_time','Sorting Time':'sorting_time'},axis=1)
dt


# In[7]:


x = dt.delivery_time
y = dt.sorting_time
plt.scatter(x,y)
plt.xlabel=("delivery_time")
plt.ylabel=("sorting_time")


# In[8]:


dt.boxplot()


# In[9]:


sns.pairplot(dt)


# In[10]:


sns.distplot(dt['delivery_time'])


# In[11]:


sns.distplot(dt['sorting_time'])


# In[12]:


sns.regplot(x='delivery_time', y='sorting_time', data=dt)


# In[13]:


model=smf.ols("sorting_time~delivery_time ", data=dt).fit()
model.summary() #build the models


# In[14]:


model.params


# In[15]:


print(model.tvalues,'\n' ,model.pvalues)


# In[16]:


(model.rsquared,model.rsquared_adj)


# #Transformation models

# In[17]:


model2 = smf.ols("np.log(sorting_time)~delivery_time", data=dt).fit() 
model2.params
model2.summary()   #build log transformation model to increase r-squared value.


# In[18]:


(model2.rsquared,model2.rsquared_adj)


# In[19]:


model3 = smf.ols("np.sqrt(sorting_time)~delivery_time", data=dt).fit() 
model3.params
model3.summary()     


# In[20]:


(model3.rsquared,model3.rsquared_adj)


# #Prediction

# In[21]:


newdata=pd.Series([10,5])   #predict the new values


# In[22]:


data_pred=pd.DataFrame(newdata, columns=['delivery_time'])
data_pred


# In[23]:


model3.predict(data_pred)


# #Inference

# Hence, np.sqrt transforfation model is suitable accuracy for the data.

# In[ ]:




