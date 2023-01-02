#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import datasets  
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing


# In[2]:


data=pd.read_csv('Fraud_check.csv')
data


# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


sns.pairplot(data)


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


data


# In[11]:


data['Status'].unique()


# In[12]:


label_encoder = preprocessing.LabelEncoder()
data['Status']= label_encoder.fit_transform(data['Status'])
data


# In[13]:


x=data.iloc[:,0:4]
y=data['Status']


# In[14]:


data['Status'].unique() 


# In[15]:


data.Status.value_counts()


# In[16]:


colnames = list(data.columns)
colnames


# In[17]:


x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)          # Splitting data into training and testing data set


# #Building Decision Tree Classifier using Entropy Criteria

# In[18]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train) 


# In[19]:


tree.plot_tree(model);             #Plot the decision tree


# In[20]:


fn=['Undergrad'	,'Marital.Status','Taxable.Income','City.Population']
cn=['Good', 'Risky']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);


# In[21]:


model.feature_importances_


# In[22]:


feature_imp = pd.Series(model.feature_importances_,index=fn).sort_values(ascending=False) 
feature_imp


# In[23]:


sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()


# In[24]:


#Predicting on test data
preds = model.predict(x_test)    
pd.Series(preds).value_counts() 


# In[25]:


preds


# In[26]:


pd.crosstab(y_test,preds)


# In[27]:


np.mean(preds==y_test)


# #Building Decision Tree Classifier (CART) using Gini Criteria

# In[28]:


model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)


# In[29]:


model_gini.fit(x_train, y_train)


# In[30]:


pred=model.predict(x_test)
np.mean(preds==y_test)     


# In[31]:


model.feature_importances_


# #Decision Tree Regression

# In[32]:


from sklearn.tree import DecisionTreeRegressor


# In[33]:


array = data.values
X = array[:,0:4]
y = array[:,3]


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# In[35]:


model = DecisionTreeRegressor()
model.fit(X_train, y_train)


# In[36]:


model.score(X_test,y_test) 


# In[ ]:




