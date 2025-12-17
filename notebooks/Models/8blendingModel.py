import os, sys
import numpy as np
import pandas as pd

from config_local import local_config

from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

out_path_XGB = os.path.join(local_config.SUBMISSIONS_DIR, "xgboost_Model.csv")
out_path_LGB = os.path.join(local_config.SUBMISSIONS_DIR, "lightgbm_Model.csv")
out_path_CAT = os.path.join(local_config.SUBMISSIONS_DIR, "catboost_Model.csv")

xgb_pred = pd.read_csv(out_path_XGB)
lgb_pred = pd.read_csv(out_path_LGB)
cat_pred = pd.read_csv(out_path_CAT)

# Ensure Id alignment is perfect
assert xgb_pred["Id"].equals(lgb_pred["Id"])
assert xgb_pred["Id"].equals(cat_pred["Id"])

# ----- WEIGHTS -----
w_xgb = 2
w_lgb = 0.5
w_cat = 1

# Blend predictions
blend = xgb_pred.copy()
blend["SalePrice"] = (
    w_xgb * xgb_pred["SalePrice"]
    + w_lgb * lgb_pred["SalePrice"]
    + w_cat * cat_pred["SalePrice"]
) / (w_xgb + w_lgb + w_cat)
# Save blended file
out_path_BLEND = os.path.join(local_config.SUBMISSIONS_DIR, "blend_xgb_lgb_cat_Model.csv")
blend.to_csv(out_path_BLEND, index=False)

# 
