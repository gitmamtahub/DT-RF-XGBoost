#!/usr/bin/env python
# coding: utf-8

# In[8]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[10]:


#import data,model and metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[11]:


#load data
iris=load_iris()
x=iris.data
y=iris.target


# In[12]:


#split data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[13]:


# classifier
clf=DecisionTreeClassifier(max_depth=3,random_state=42)
clf.fit(x_train,y_train)


# In[14]:


# prediction
y_predict=clf.predict(x_test)
y_predict


# In[15]:


y_test


# In[16]:


# Metrics Measures
print("Accuracy :", accuracy_score(y_test,y_predict))
print("Report :/n", classification_report(y_test,y_predict))


# In[28]:


# Random-Forest : import libraries, data, model and metrics
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[29]:


# load data
california=fetch_california_housing()
x=california.data
y=california.target


# In[30]:


california


# In[31]:


x[:10]


# In[32]:


y[:10]


# In[33]:


# split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=0)


# In[35]:


x_train[:5]


# In[36]:


y_train[:5]


# In[37]:


# modeling
rf=RandomForestRegressor(n_estimators=100,random_state=0)
rf.fit(x_train,y_train)


# In[41]:


y_pred=rf.predict(x_test)
y_pred


# In[42]:


y_test


# In[43]:


print("RMSE :",np.sqrt(mean_squared_error(y_test,y_pred)))
print("R2-Score :",r2_score(y_test,y_pred))


# In[63]:


# XGBoost - import libraries, models, data and metrics
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[64]:


# load data
diabetes=load_diabetes()
x=diabetes.data
y=diabetes.target


# In[65]:


diabetes


# In[66]:


x[:5]


# In[67]:


y[:5]


# In[68]:


# Convert targets to binary classification
y = [1 if target > 140 else 0 for target in y]


# In[69]:


# split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)


# In[70]:


y_train[:5]


# In[71]:


y_test[:5]


# In[72]:


# Convert the dataset into an optimized data structure called Dmatrix that XGBoost supports
D_train = xgb.DMatrix(x_train, label=y_train)
D_test = xgb.DMatrix(x_test, label=y_test)


# In[73]:


# Define the parameters for the XGBoost classifier
param = {
    'eta': 0.3, 
    'max_depth': 3,
    'objective': 'multi:softprob',  
    'num_class': 2}  

steps = 20  # The number of training iterations


# In[74]:


# model building
model=xgb.train(param,D_train,steps)


# In[76]:


#prediction
pred=model.predict(D_test)
pred


# In[77]:


# softmax to class
y_pred=np.asarray([np.argmax(pre) for pre in pred])


# In[78]:


y_pred


# In[80]:


# Metrics Measures
print("Accuracy :", accuracy_score(y_test,y_pred))
print("Report :/n", classification_report(y_test,y_pred))


# In[ ]:




