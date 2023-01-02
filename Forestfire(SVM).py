#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


# In[2]:


df=pd.read_csv('forestfires.csv')
df


# In[3]:


df.info()


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.duplicated()


# In[7]:


sns.pairplot(df)


# In[8]:


sns.heatmap(df.isnull(),cmap='Reds')


# In[9]:


sns.boxplot(data=df)


# In[10]:


df1=df.iloc[:,2:]
df1


# In[11]:


array = df1.values
X = array[:,0:28]
Y = array[:,28]


# In[12]:


X


# In[13]:


Y


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3)


# In[15]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# #Grid Search CV

# ###rbf

# In[16]:


clf = SVC()
param_grid = [{'kernel':['rbf'],'gamma':[50,5,10,0.5],'C':[15,14,13,12,11,10,0.1,0.001] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X_train,y_train)


# In[17]:


gsv.best_params_ , gsv.best_score_ 


# In[18]:


clf = SVC(C= 15, gamma = 50)
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred)


# ###Linear

# In[19]:


clf = SVC(kernel= "linear") 
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred) 


# In[20]:


clf = SVC(kernel= "poly") 
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred) 

