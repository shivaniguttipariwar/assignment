#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels import formula
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.formula.api as smf


# In[6]:


data = pd.read_csv('50_Startups.csv')
data


# In[7]:


data.describe()


# In[8]:


data.info()


# In[9]:


#correlation
data.corr()


# In[10]:


sns.pairplot(data)


# In[11]:


sns.distplot(data['Profit'])


# In[12]:


data = data.rename({'R&D Spend':'RD_spend','Marketing Spend':'Marketing_Spend'},axis=1)
data


# In[13]:


data.drop('State',axis=1)


# In[14]:


model = smf.ols("Profit~RD_spend+Administration+Marketing_Spend+Profit",data=data).fit()
model.summary()


# In[15]:


model.params


# In[16]:


print(model.tvalues, '\n', model.pvalues)


# In[17]:


(model.rsquared,model.rsquared_adj)


# In[18]:


md= smf.ols("Profit~RD_spend",data=data).fit()
print(md.tvalues, '\n' , md.pvalues)


# In[19]:


md= smf.ols("Profit~Administration",data=data).fit()
print(md.tvalues, '\n' , md.pvalues)


# In[20]:


md= smf.ols("Profit~RD_spend+Administration",data=data).fit()
md.summary()


# In[21]:


rsq_RD = smf.ols("RD_spend~Marketing_Spend+Administration",data=data).fit().rsquared
vif_RD = 1/(1-rsq_RD) 
rsq_A = smf.ols("Administration~RD_spend+Marketing_Spend",data=data).fit().rsquared  
vif_A= 1/(1-rsq_A) 
rsq_M= smf.ols("Marketing_Spend~Administration+RD_spend",data=data).fit().rsquared  
vif_M = 1/(1-rsq_M) 
d1={'Variables':['Administration','RD_spend','Marketing_Spend'],'VIF':[vif_A,vif_RD,vif_M]}
vif_frame = pd.DataFrame(d1)
vif_frame


# In[22]:


import statsmodels.api as sm
qqplot=sm.qqplot(model.resid,line='q') 
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[23]:


def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std()


# In[24]:


plt.scatter(get_standardized_values(model.fittedvalues),
            get_standardized_values(model.resid))
plt.title('Residual Plot')
plt.xlabel('Standardized Fitted values')
plt.ylabel('Standardized residual values')
plt.show()


# In[25]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Administration", fig=fig)
plt.show()


# In[26]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "RD_spend", fig=fig)
plt.show()


# In[27]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Marketing_Spend", fig=fig)
plt.show()


# In[28]:


model_influence = model.get_influence()
(c, _) = model_influence.cooks_distance
c


# In[29]:


fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(data)), np.round(c, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[30]:


(np.argmax(c),np.max(c))


# In[31]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model)
plt.show()


# In[32]:


k = data.shape[1]
n = data.shape[0]
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff


# In[33]:


data[data.index.isin([47, 49])]


# In[34]:


data_new=data.drop(data.index[[47,49]],axis=0).reset_index()


# In[35]:


data_new=data_new.drop(['index'],axis=1)


# In[36]:


data_new


# In[37]:


final_Newdata= smf.ols('Profit~Administration+Marketing_Spend',data =data_new).fit()


# In[38]:


(final_Newdata.rsquared,final_Newdata.aic)


# In[39]:


final_Newdata= smf.ols('Profit~RD_spend+Marketing_Spend',data =data_new).fit()


# In[40]:


(final_Newdata.rsquared,final_Newdata.aic)


# In[41]:


new_data=pd.DataFrame({'Adiministration':100,'RD_spend':150,'Marketing_Spend':200},index=[1])
new_data


# In[42]:


final_Newdata.predict(new_data)


# In[ ]:




