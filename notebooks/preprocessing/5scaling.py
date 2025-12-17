#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import config_local.local_config as local_config

from sklearn.preprocessing import StandardScaler


# In[2]:


def is_continuous(series, threshold=0.05):
    if series.dtype not in ['int64', 'float64']:
        return False  # not numeric
    
    ratio = series.nunique() / len(series)
    return ratio > threshold


# In[3]:


train = pd.read_csv(local_config.TRAIN_PROCESS4_CSV)
test = pd.read_csv(local_config.TEST_PROCESS4_CSV)


# In[4]:


X_train = train.drop(columns=["logSP"]).copy()
y = train["logSP"]
X_test  = test.copy()

X_rows = len(X_train)

train_numeric = X_train.select_dtypes(include='number')
train_numeric_cont = [col for col in train_numeric.columns if is_continuous(train_numeric[col])] 
train_numeric_desc = [col for col in train_numeric.columns if col not in train_numeric_cont] 

scaler = StandardScaler()

for col in X_train.columns:
    if col in train_numeric_cont:
        X_train[col] = scaler.fit_transform(X_train[[col]])
        X_test[col]  = scaler.transform(X_test[[col]])



# In[5]:


X_train = pd.concat([X_train, y], axis = 1)


# In[6]:


from pathlib import Path

# Ensure directories exist
Path(local_config.TRAIN_PROCESS5_CSV).parent.mkdir(parents=True, exist_ok=True)
Path(local_config.TEST_PROCESS5_CSV).parent.mkdir(parents=True, exist_ok=True)

X_train.to_csv(local_config.TRAIN_PROCESS5_CSV, index=False)  
X_test.to_csv(local_config.TEST_PROCESS5_CSV, index=False)  


# In[7]:


train_cols = set(train.columns)
test_cols  = set(test.columns)


# In[8]:


common = train_cols.intersection(test_cols)
print("Common columns:", common)


# In[9]:


only_train = train_cols - test_cols
print("Columns only in train:", only_train)


# In[10]:


only_test = test_cols - train_cols
print("Columns only in test:", only_test)

