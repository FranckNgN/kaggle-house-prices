#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import config_local.local_config as local_config

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[2]:


def add_kmeans(
    train,
    test,
    k=4,
    cols=("GrLivArea", "TotalBsmtSF", "1stFlrSF", "GarageCars", "YearBuilt", "OverallQual"),
    seed=42,
):
    cols = [c for c in cols if c in train.columns and c in test.columns]
    X = pd.concat([train[cols], test[cols]]).to_numpy()
    labels = KMeans(k, n_init=20, random_state=seed)\
        .fit_predict(StandardScaler().fit_transform(X))
    
    train["KMeansCluster"] = labels[:len(train)].astype("int16")
    test["KMeansCluster"]  = labels[len(train):].astype("int16")

    return train, test


# In[3]:


train = pd.read_csv(local_config.TRAIN_PROCESS3_CSV)  
test = pd.read_csv(local_config.TEST_PROCESS3_CSV)


# In[4]:


to_drop = []#['Street']
train.drop(columns=to_drop, inplace=True, axis=1)


# In[5]:


for df in (train, test):
    df["Age"] = df["YrSold"] - df["YearBuilt"]
    df["Garage_Age"] = df["YrSold"] - df["GarageYrBlt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]


# # K-means clustering

# In[6]:


kmeans_cols = [
    "GrLivArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "GarageCars",
    "YearBuilt",
    "OverallQual",
]
train, test = add_kmeans(train, test, cols=kmeans_cols)


# In[7]:


from pathlib import Path

# Ensure directories exist
Path(local_config.TRAIN_PROCESS4_CSV).parent.mkdir(parents=True, exist_ok=True)
Path(local_config.TEST_PROCESS4_CSV).parent.mkdir(parents=True, exist_ok=True)

train.to_csv(local_config.TRAIN_PROCESS4_CSV, index=False)  
test.to_csv(local_config.TEST_PROCESS4_CSV, index=False) 

