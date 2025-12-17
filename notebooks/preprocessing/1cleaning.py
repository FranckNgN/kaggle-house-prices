#!/usr/bin/env python
# coding: utf-8

# ## Cleaning columns

# In[1]:


import os, sys
import pandas as pd

from config_local import local_config


# In[2]:


train = pd.read_csv(local_config.TRAIN_CSV, index_col="Id")#, index_col="Id"
test  = pd.read_csv(local_config.TEST_CSV, index_col="Id")#, index_col="Id"

print(f"Train shape: {train.shape}  |  Test shape: {test.shape}")
print(train.head(3))


# # Missing values

# In[3]:


def fill_missing_with_none_or_zero(df):
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].replace(["NA", ""], pd.NA)   # normalize weird missing codes
            df[col] = df[col].fillna("<None>")
    return df


missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
missing


# In[4]:


train_filled = fill_missing_with_none_or_zero(train)
test_filled = fill_missing_with_none_or_zero(test)


# In[5]:


def summarize_columns(df, max_unique=15):
    summary = {}
    for col in df.columns:
        col_type = df[col].dtype
        uniques = df[col].dropna().unique()
        n_unique = len(uniques)

        if pd.api.types.is_numeric_dtype(df[col]):
            if n_unique <= max_unique:
                summary[col] = {
                    "type": "numeric (discrete)",
                    "unique_values": sorted(uniques)
                }
            else:
                summary[col] = {
                    "type": "numeric (continuous)",
                    "unique_values": f"{n_unique} unique values"
                }
        else:
            if n_unique <= max_unique:
                summary[col] = {
                    "type": "categorical",
                    "unique_values": uniques.tolist()
                }
            else:
                summary[col] = {
                    "type": "categorical",
                    "unique_values": f"{n_unique} unique values"
                }
    return pd.DataFrame(summary).T

# Usage
feature_summary = summarize_columns(train_filled)
print(feature_summary.head(20))

from pathlib import Path

# Ensure directory exists
Path(local_config.FEATURE_SUMMARY_CSV).parent.mkdir(parents=True, exist_ok=True)
feature_summary.to_csv(local_config.FEATURE_SUMMARY_CSV, index=True)


# In[6]:


from pathlib import Path

# Ensure directories exist
Path(local_config.TRAIN_PROCESS1_CSV).parent.mkdir(parents=True, exist_ok=True)
Path(local_config.TEST_PROCESS1_CSV).parent.mkdir(parents=True, exist_ok=True)

train_filled.to_csv(local_config.TRAIN_PROCESS1_CSV, index=False)
test_filled.to_csv(local_config.TEST_PROCESS1_CSV, index=False)

