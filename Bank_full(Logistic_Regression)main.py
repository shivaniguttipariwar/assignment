#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# In[2]:


bank=pd.read_csv('bank-full.csv',sep =';' )
bank.head()


# In[3]:


bank.info()


# In[4]:


bank.shape


# In[5]:


bank[categorical].isnull().sum()


# ###Factorization

# 

# In[6]:


bank[['job','marital','education','default','housing','loan','contact','month','poutcome','y']]=bank[['job','marital','education','default','housing','loan','contact','month','poutcome','y']].apply(lambda x: pd.factorize(x)[0])
bank               #converting into dummy variables


# In[7]:


X = bank.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
Y = bank.iloc[:,16]
classifier = LogisticRegression()
classifier.fit(X,Y) 


# In[8]:


classifier.coef_  # coefficients of features   


# In[9]:


classifier.predict_proba (X) # Probability values   


# # Prediction

# In[10]:


y_pred = classifier.predict(X)
bank["y_pred"] = y_pred
bank


# In[11]:


y_prob = pd.DataFrame(classifier.predict_proba(X.iloc[:,:]))
new_df = pd.concat([bank,y_prob],axis=1)
new_df  


# ##confusion matrix

# In[12]:


confusion_matrix = confusion_matrix(Y,y_pred)
print (confusion_matrix) 


# In[13]:


pd.crosstab(y_pred,Y)  


# In[14]:


#type(y_pred)
accuracy = sum(Y==y_pred)/bank.shape[0]
accuracy


# In[15]:


print (classification_report (Y, y_pred))  


# In[16]:


Logit_roc_score=roc_auc_score(Y,classifier.predict(X))
Logit_roc_score                                   # logistic ROC score 


# ###ROC_Curve

# In[17]:


fpr, tpr, thresholds = roc_curve(Y,classifier.predict_proba(X)[:,1]) 
plt.plot(fpr, tpr, label='Logistic Regression (area=%0.2f)'% Logit_roc_score)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])                 
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')    
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()                                                      #fpr, tpr, thresholds = precision-recall_curve(Y,classifier.predict_proba(X)[:,1]) 


# In[18]:


y_prob1 = pd.DataFrame(classifier.predict_proba(X)[:,1]) 
y_prob1                              


# In[ ]:





# In[ ]:





# In[ ]:




