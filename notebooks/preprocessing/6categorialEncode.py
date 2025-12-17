#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import config_local.local_config as local_config
from pathlib import Path

# Load datasets
train = pd.read_csv(local_config.TRAIN_PROCESS5_CSV)  # has logSP only
test  = pd.read_csv(local_config.TEST_PROCESS5_CSV)

# Target
y = train["logSP"].copy()

# Features only (drop target)
X_train = train.drop(columns=["logSP"]).copy()
X_test  = test.copy()

# Concat so dummies align
all_X = pd.concat([X_train, X_test], axis=0, ignore_index=True)

# One-hot encode categoricals
all_X_enc = pd.get_dummies(all_X, drop_first=True, dtype="int8")

# One-hot encode categoricals *only* (optional but cleaner)
cat_cols = all_X.select_dtypes(include=["object"]).columns

# Split back into train/test
X_train_enc = all_X_enc.iloc[:len(X_train), :].copy()
X_test_enc  = all_X_enc.iloc[len(X_train):, :].copy()


train_enc = pd.concat([y.reset_index(drop=True), X_train_enc], axis=1)
all_X_enc = pd.get_dummies(all_X, columns=cat_cols, drop_first=True, dtype="int8")

# Ensure directories exist
Path(local_config.TRAIN_PROCESS6_CSV).parent.mkdir(parents=True, exist_ok=True)
Path(local_config.TEST_PROCESS6_CSV).parent.mkdir(parents=True, exist_ok=True)

# Save
train_enc.to_csv(local_config.TRAIN_PROCESS6_CSV, index=False)
X_test_enc.to_csv(local_config.TEST_PROCESS6_CSV, index=False)


# # Run ANOVA on categorial variables to check wether there is variance
