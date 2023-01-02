#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn import preprocessing


# In[2]:


data=pd.read_csv('Fraud_check.csv')
data


# In[3]:


data.info()


# In[4]:


sns.pairplot(data)


# In[5]:


sns.heatmap(data.isnull(),cmap='Reds')


# In[6]:


plt.figure(figsize=(20,10))
sns.heatmap(data.corr(),annot=True)


# In[7]:


label_encoder = preprocessing.LabelEncoder()
data['Undergrad']= label_encoder.fit_transform(data['Undergrad'])
data['Urban']= label_encoder.fit_transform(data['Urban'])
data['Marital.Status']= label_encoder.fit_transform(data['Marital.Status'])


# In[8]:


data


# In[9]:


data['Status'] = data['Taxable.Income'].apply(lambda Income: 'Risky' if Income <= 30000 else 'Good')


# In[10]:


data['Status']= label_encoder.fit_transform(data['Status'])


# In[11]:


data


# In[12]:


data.Status.unique()


# In[13]:


x=data.iloc[:,0:4]
y=data['Status']


# #Bagged Decision Trees for Classification

# In[14]:


num_trees = 100
seed=8
kfold = KFold(n_splits=100, shuffle = True, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, x,y, cv=kfold)
print(results.mean())


# #Stacking Ensemble for Classification

# In[15]:


kfold = KFold(n_splits=10,shuffle=True, random_state=8)
estimators = []
model1 = LogisticRegression(max_iter=100)                          # create the sub models
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
ensemble = VotingClassifier(estimators)                            # create the ensemble model
results = cross_val_score(ensemble, x, y, cv=kfold)
print(results.mean())


# #Random Forest Classification

# In[16]:


num_trees = 100
max_features = 3
kfold = KFold(n_splits=10, shuffle= True ,random_state=8)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, x, y, cv=kfold)
print(results.mean())


# #Boost Classification

# In[17]:


num_trees = 100
seed=8
kfold = KFold(n_splits=100, shuffle = True, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, x,y, cv=kfold)
print(results.mean())


# In[ ]:





# In[ ]:




