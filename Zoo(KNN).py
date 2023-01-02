#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


zoo=pd.read_csv('Zoo.csv')


# In[3]:


zoo


# In[4]:


zoo.shape


# In[5]:


zoo.info()


# In[6]:


zoo.duplicated()


# In[7]:


sns.pairplot(zoo)


# In[8]:


sns.heatmap(zoo.isnull(),cmap='Blues')


# In[9]:


zoo=zoo.drop("animal name",axis=1)


# In[10]:


array = zoo.values
X = array[:, 0:16]
Y = array[:, 16]


# In[11]:


X


# In[12]:


Y


# In[13]:


num_folds = 20
kfold = KFold(n_splits=20)


# In[14]:


model = KNeighborsClassifier(n_neighbors=20)
results = cross_val_score(model, X, Y, cv=kfold)


# In[15]:


print(results.mean())


# #Grid Search for Algorithm Tuning

# In[16]:


from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy 


# In[17]:


n_neighbors = numpy.array(range(1,40))
param_grid = dict(n_neighbors=n_neighbors)


# In[18]:


model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)


# In[19]:


print(grid.best_score_)
print(grid.best_params_)


# #Visualizing the CV results

# In[20]:


import warnings
warnings.filterwarnings('ignore')


# In[21]:


import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
# choose k between 1 to 41
k_range = range(1, 41)
k_scores = []
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, Y, cv=5)
    k_scores.append(scores.mean())
# plot to see clearly
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

