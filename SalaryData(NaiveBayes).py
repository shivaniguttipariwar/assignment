#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install naive-bayes')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB


# In[3]:


train=pd.read_csv('SalaryData_Train.csv')
train


# In[4]:


test=pd.read_csv('SalaryData_Test.csv')
test


# In[5]:


train.info()


# In[6]:


test.info()


# In[7]:


test.describe().round(2).style.background_gradient(cmap = 'Reds')


# In[8]:


train.describe().round(2).style.background_gradient(cmap = 'Blues')


# In[9]:


correlation = test.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')       
plt.title('Correlation between different fearures')


# In[10]:


correlation = train.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')       
plt.title('Correlation between different fearures')


# In[11]:


sns.heatmap(test.isnull(),cmap='Reds')


# In[12]:


sns.heatmap(train.isnull(),cmap='Blues')


# In[13]:


sns.pairplot(train)


# In[14]:


sns.pairplot(test)


# In[15]:


train[['workclass','education','maritalstatus','occupation','relationship','race','sex','native','Salary']] = train[['workclass','education','maritalstatus','occupation','relationship','race','sex','native','Salary']].apply(lambda x: pd.factorize(x)[0])
train


# In[16]:


test[['workclass','education','maritalstatus','occupation','relationship','race','sex','native','Salary']] = test[['workclass','education','maritalstatus','occupation','relationship','race','sex','native','Salary']].apply(lambda x: pd.factorize(x)[0])
test


# In[17]:


X=train.iloc[:,0:13]
Y=train.iloc[:,13]
x=test.iloc[:,0:13]
y=test.iloc[:,13]


# In[18]:


X


# In[19]:


Y


# In[20]:


x


# In[21]:


y


# #Naive Bayes

# ###Multinominal Naive Bayes

# In[22]:


classifier_mb = MB()
classifier_mb.fit(X,Y)


# In[23]:


train_pred_m = classifier_mb.predict(X)
accuracy_train_m = np.mean(train_pred_m==Y)


# In[24]:


test_pred_m = classifier_mb.predict(x)
accuracy_test_m = np.mean(test_pred_m==y)


# In[25]:


print('Training accuracy is:',accuracy_train_m,'\n','Testing accuracy is:',accuracy_test_m)


# ###Gaussian Naive Bayes

# In[26]:


classifier_gb = GB()
classifier_gb.fit(X,Y) 


# In[27]:


train_pred_g = classifier_gb.predict(X)
accuracy_train_g = np.mean(train_pred_g==Y)


# In[28]:


test_pred_g = classifier_gb.predict(X)
accuracy_test_g = np.mean(test_pred_g==Y)


# In[29]:


print('Training accuracy is:',accuracy_train_g,'\n','Testing accuracy is:',accuracy_test_g)

