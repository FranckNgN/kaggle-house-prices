#!/usr/bin/env python
# coding: utf-8

# Looking at the skew and kurtosis of sale price

# In[1]:


import pandas as pd
import numpy as np

from scipy.stats import boxcox
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

import matplotlib.pyplot as plt

from config_local import local_config  # provides TRAIN_CSV, TEST_CSV, SUBMISSIONS_DIR


# In[2]:


def is_continuous(series, threshold=0.05):
    if series.dtype not in ['int64', 'float64']:
        return False  # not numeric
    
    ratio = series.nunique() / len(series)
    return ratio > threshold


# In[3]:


train = pd.read_csv(local_config.TRAIN_PROCESS2_CSV)   # e.g. data/train_filled.csv           
test = pd.read_csv(local_config.TEST_PROCESS2_CSV)    # e.g. data/test_filled.csv


# # Dealing with skewness 

# In[4]:


# numeric columns
num_cols = lambda train, threshold=0.05: [col for col in train.columns if is_continuous(train[col], threshold)]

# skew before
skew_before = train[num_cols].apply(lambda x: skew(x.dropna()))
skewed_cols = skew_before[skew_before.abs() > 0.75].index

print("Skewed columns (|skew| > 0.75):")
print(list(skewed_cols))#TODO: Check for skewed categorial numerical if htere is a point int unskweing

# transform
pt = PowerTransformer(method="yeo-johnson")
train[skewed_cols] = pt.fit_transform(train[skewed_cols])
test[skewed_cols]  = pt.transform(test[skewed_cols])


# In[5]:


from pathlib import Path

# Ensure directories exist
Path(local_config.TRAIN_PROCESS3_CSV).parent.mkdir(parents=True, exist_ok=True)
Path(local_config.TEST_PROCESS3_CSV).parent.mkdir(parents=True, exist_ok=True)

train.to_csv(local_config.TRAIN_PROCESS3_CSV, index=False)
test.to_csv(local_config.TEST_PROCESS3_CSV, index=False)

