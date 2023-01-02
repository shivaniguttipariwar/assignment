#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[7]:


df=pd.read_excel("CocaCola_Sales_Rawdata.xlsx")
df


# In[8]:


df.isnull().sum()


# In[9]:


sns.lineplot(x="Quarter",y="Sales",data=df)


# In[14]:


import statsmodels.api as smf
seasonal_ts_add=smf.tsa.seasonal_decompose(df["Sales"],period=10)
seasonal_ts_add.plot()


# In[15]:


sns.boxplot(data=df['Sales'])


# In[16]:


quarter =['Q1','Q2','Q3','Q4']


# In[17]:


p = df["Quarter"][0]
p[0:2]
df['quarter']= 0

for i in range(42):
    p = df["Quarter"][i]
    df['quarter'][i]= p[0:2]

df.head()


# In[18]:


quarter_dummies = pd.DataFrame(pd.get_dummies(df['quarter']))
df1 = pd.concat([df,quarter_dummies],axis = 1)
df1


# In[19]:


df1["t"] = np.arange(1,43)
df1["t_squared"] = df1["t"]*df1["t"]
df1["log_Sales"] = np.log(df1["Sales"])
df1


# In[20]:


Train = df1.head(30)
Test = df1.tail(10)


# In[21]:


import statsmodels.formula.api as smf
linear_model = smf.ols('Sales~t',data=Train).fit()                    #Llinear
pred_linear = pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
print("RMSE Linear: ",rmse_linear)


# In[22]:


Exp = smf.ols('log_Sales~t',data=Train).fit()                        #Exponential
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
print("RMSE Exponential: ",rmse_Exp)


# In[23]:


Quad = smf.ols('Sales~t+t_squared',data=Train).fit()                 #Quadratic 
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
print("RMSE Quadratic: ",rmse_Quad)


# In[24]:


add_sea = smf.ols('Sales~Q1+Q2+Q3+Q4',data=Train).fit()              #Additive seasonality
pred_add_sea = pd.Series(add_sea.predict(Test[['Q1', 'Q2', 'Q3', 'Q4']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
print("RMSE Additive seasonality: ",rmse_add_sea)


# In[25]:


add_sea_Quad = smf.ols('Sales~t+t_squared+Q1+Q2+Q3+Q4',data=Train).fit()        #Additive Seasonality Quadratic
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Q1', 'Q2', 'Q3', 'Q4','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
print("RMSE Additive Seasonality Quadratic:",rmse_add_sea_quad )


# In[26]:


Mul_sea = smf.ols('log_Sales~Q1+Q2+Q3+Q4',data = Train).fit()          # Multiplicative Seasonality
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
print("RMSE Multiplicative Seasonality:",rmse_Mult_sea)


# In[27]:


Mul_Add_sea = smf.ols('log_Sales~t+Q1+Q2+Q3+Q4',data = Train).fit()    #Multiplicative Additive Seasonality
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
print("RMSE Multiplicative Additive Seasonality:",rmse_Mult_add_sea )


# In[28]:


data1 = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),         #Testing
        "RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}              
table_rmse=pd.DataFrame(data1)
table_rmse


# In[29]:


data = [['Q3_96', 'Q3'], ['Q4_96', 'Q4'], ['Q1_97', 'Q1'],['Q2_97', 'Q2']]
print(data)
forecast = pd.DataFrame(data, columns = ['Quarter', 'quarter'])
forecast


# In[30]:


dummies = pd.DataFrame(pd.get_dummies(forecast['quarter']))         # Create dummies and T and T-Squared columns
forecast1 = pd.concat([forecast,dummies],axis = 1)
forecast1["t"] = np.arange(1,5)   
forecast1["t_squared"] = forecast1["t"]*forecast1["t"] 
print("\nAfter Dummy, T and T-Square\n\n",forecast1.head())


# In[31]:


model_full = smf.ols('Sales~t+t_squared+Q1+Q2+Q3+Q4',data=df1).fit()        # Forecasting using Additive Seasonality Quadratic Model
pred_new  = pd.Series(model_full.predict(forecast1))
pred_new
forecast1["forecasted_sales"] = pd.Series(pred_new)


# In[32]:


Final_predict = forecast1.loc[:, ['Quarter', 'forecasted_sales']]           # Final Prediction for next 4 Quarters
Final_predict


# In[ ]:




