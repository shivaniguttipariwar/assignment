#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf


# In[9]:


salary=pd.read_csv('Salary_Data.csv')
salary


# In[10]:


salary.describe()


# In[11]:


salary.corr()


# In[12]:


x=salary.YearsExperience
y=salary.Salary
plt.scatter(x,y)
plt.xlabel('YearsExperience')
plt.ylabel('Salary')


# In[13]:


sns.distplot(salary['YearsExperience'])


# In[14]:


sns.distplot(salary['Salary'])
             


# In[15]:


sns.regplot(x="YearsExperience", y="Salary", data=salary);


# In[16]:


import statsmodels.formula.api as smf
model = smf.ols("Salary~YearsExperience",data = salary).fit()
model.summary()


# In[17]:


pred=model.params


# In[18]:


print(model.tvalues, '\n', model.pvalues)    


# In[19]:


(model.rsquared,model.rsquared_adj)


# In[20]:


newsalary=pd.Series([30,40])


# In[21]:


data_pred=pd.DataFrame(newsalary,columns=['YearsExperience'])
data_pred


# In[22]:


model.predict(data_pred)


# In[ ]:





# In[ ]:





# In[ ]:




