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


glass=pd.read_csv('glass.csv')


# In[3]:


glass


# In[4]:


glass.info()


# In[5]:


glass.duplicated()


# In[6]:


sns.pairplot(glass)


# In[7]:


sns.heatmap(glass.isnull(),cmap='bone')


# In[8]:


array = glass.values
X = array[:, 0:9]
Y = array[:, 9]


# In[9]:


num_folds = 70
kfold = KFold(n_splits=70)


# In[10]:


model = KNeighborsClassifier(n_neighbors=50)
results = cross_val_score(model, X, Y, cv=kfold)


# In[11]:


print(results.mean())


# #Grid Search for Algorithm Tuning

# In[12]:


from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy 


# In[13]:


n_neighbors = numpy.array(range(1,40))
param_grid = dict(n_neighbors=n_neighbors)


# In[14]:


model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)


# In[15]:


print(grid.best_score_)
print(grid.best_params_)


# #Visualizing the CV results

# In[16]:


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

