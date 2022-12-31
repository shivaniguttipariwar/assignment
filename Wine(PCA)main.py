#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sn


# In[2]:


wine=pd.read_csv('wine.csv')
wine


# In[3]:


wine.head()


# In[4]:


wine.info()


# In[5]:


sn.pairplot(wine)


# ###Normalization

# In[6]:


wine.data = wine.iloc[:,1:]
data =wine.data.values
data


# In[7]:


from sklearn.preprocessing import scale
wine_normal = scale(data)
wine_normal


# #PCA

# In[8]:


pca = PCA(n_components = 13)
pca_values = pca.fit_transform(wine_normal)
pca_values 


# In[9]:


pca.components_           #loading or weight


# In[10]:


var = pca.explained_variance_ratio_                  # The amount of variance that each PCA explains is 
var


# In[11]:


var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1                                                    # Cumulative variance 


# In[12]:


plt.plot(var1,color="red")                              # Variance plot for PCA components obtained 


# In[13]:


pca_values[:,0:1]


# In[14]:


final_df = pd.concat([pd.DataFrame(pca_values[:,0:3],columns=['pc1','pc2','pc3']),wine['Type']], axis=1)
final_df


# ###Visualization

# In[15]:


sn.scatterplot(data=final_df,x='pc1',y='pc2',hue='Type',s = 100)  


# In[16]:


p1 = sn.scatterplot(data=final_df,x='pc1',y='pc2',s = 100)  
for line in range(0,final_df.shape[0]):
     p1.text(final_df.pc1[line], final_df.pc2[line], final_df.Type[line], horizontalalignment='left', size='medium')


# #Clustering Algorithms

# In[17]:


from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import warnings 
warnings.filterwarnings('ignore')


# In[18]:


get_ipython().system('pip3 install KMeans')


# ##HIERARCHAICAL Clustering

# In[19]:


p = np.array(wine_normal) 
z = linkage(wine_normal, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(
    z,
)
plt.show()   


# In[20]:


h_complete = AgglomerativeClustering(n_clusters=3, linkage='complete',affinity = "euclidean").fit(wine_normal) 
cluster_labels=pd.Series(h_complete.labels_)
cluster_labels
wine['clust']=cluster_labels # creating a  new column and assigning it to new column 
wine


# In[21]:


data = wine[(wine.clust==3)]
data  


# ###K-Means Clustering

# Elbow-curve

# In[22]:


fig = plt.figure(figsize=(10, 8))
WCSS = []
for i in range(1, 11):
    clf = KMeans(n_clusters=i)
    clf.fit(wine_normal)
    WCSS.append(clf.inertia_) 
plt.plot(range(1, 11), WCSS)
plt.title('The Elbow Method')
plt.ylabel('WCSS')
plt.xlabel('Number of Clusters')
plt.show()  


# In[23]:


WCSS


# k=3

# In[24]:


clf = KMeans(n_clusters=3)
y_kmeans = clf.fit_predict(wine_normal)  


# In[25]:


y_kmeans
clf.labels_ 


# In[26]:


clf.cluster_centers_ 


# In[27]:


clf.inertia_


# In[28]:


md=pd.Series(y_kmeans)  
wine['clust']=md 
wine


# In[29]:


WCSS


# In[ ]:




