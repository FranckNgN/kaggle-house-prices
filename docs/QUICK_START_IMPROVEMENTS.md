# Quick Start: Model Improvements

**Current Best:** Stacking RMSE 0.112898 (Kaggle: 0.62673)

---

## 🚀 Immediate Actions (Do These First!)

### 1. Run Optimized Blending (5 minutes)
You just implemented automatic weight optimization. Test it now:

```bash
python notebooks/Models/10blendingModel.py
```

This will automatically find optimal weights for your blending model using OOF predictions.

**Expected Improvement:** 0.001-0.005 RMSE

---

### 2. Try Different Stacking Meta-Models (30 minutes)

Your stacking model currently uses Lasso. Try these alternatives:

#### Option A: Ridge Meta-Model
Edit `config_local/model_config.py`:
```python
STACKING = {
    ...
    "meta_model": "ridge",  # Change from "lasso"
    "meta_model_params": {
        "alpha": 0.1,  # Try: 0.01, 0.1, 1.0, 10.0
        "random_state": 42,
    },
    ...
}
```

#### Option B: XGBoost Meta-Model (Often Best!)
You'll need to modify `notebooks/Models/11stackingModel.py` to support XGBoost:

```python
# Around line 141, add:
elif cfg["meta_model"] == "xgboost":
    from xgboost import XGBRegressor
    meta_model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
        device="cuda"  # If you have GPU
    )
```

Then update config:
```python
"meta_model": "xgboost",
"meta_model_params": {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.05,
    "random_state": 42
}
```

**Expected Improvement:** 0.003-0.008 RMSE

---

### 3. Add Target Encoding (1-2 hours)

I've created `notebooks/preprocessing/7targetEncoding.py` for you!

**Steps:**
1. First, add process7 paths to `config_local/local_config.py`:
```python
TRAIN_PROCESS7_CSV = INTERIM_TRAIN_DIR / "train_process7.csv"
TEST_PROCESS7_CSV = INTERIM_TEST_DIR / "test_process7.csv"
```

2. Run target encoding:
```bash
python notebooks/preprocessing/7targetEncoding.py
```

3. Update your model scripts to use process7 data:
```python
# In model scripts, change:
train = pd.read_csv(local_config.TRAIN_PROCESS6_CSV)
# To:
train = pd.read_csv(local_config.TRAIN_PROCESS7_CSV)
```

4. Retrain your best models (XGBoost, LightGBM, CatBoost, Stacking)

**Expected Improvement:** 0.005-0.015 RMSE

---

### 4. Add Polynomial Features (30 minutes)

Add squared terms for key features. Create `notebooks/preprocessing/8polynomialFeatures.py`:

```python
import pandas as pd
import numpy as np
from config_local import local_config

def add_polynomial_features(train, test):
    """Add squared terms for important features."""
    key_features = [
        "TotalSF",
        "OverallQual", 
        "GrLivArea",
        "GarageArea",
        "Age"
    ]
    
    for feat in key_features:
        if feat in train.columns:
            train[f"{feat}_squared"] = train[feat] ** 2
            test[f"{feat}_squared"] = test[feat] ** 2
    
    return train, test

# Load process7, add polynomials, save as process8
```

**Expected Improvement:** 0.002-0.005 RMSE

---

## 📊 Testing Strategy

After each improvement:

1. **Retrain your best models:**
   - XGBoost (optimized)
   - LightGBM (optimized)  
   - CatBoost (optimized)
   - Stacking

2. **Compare CV scores:**
   ```python
   # Check runs/model_performance.csv
   # Look for RMSE improvements
   ```

3. **Run blending with optimized weights:**
   ```bash
   python notebooks/Models/10blendingModel.py
   ```

4. **Submit best model to Kaggle** to verify improvement

---

## 🎯 Recommended Order

**Today:**
1. ✅ Run optimized blending
2. Try Ridge meta-model for stacking
3. Test XGBoost meta-model for stacking

**This Week:**
4. Implement target encoding
5. Add polynomial features
6. Retrain all models
7. Compare results

**Next Week:**
8. Feature selection
9. Multi-level stacking
10. Advanced feature engineering

---

## 💡 Pro Tips

1. **One change at a time:** Test each improvement individually
2. **Track everything:** Keep notes in `notebooks/Journal.ipynb`
3. **Version control:** Save predictions from each experiment
4. **CV > Leaderboard:** Trust your cross-validation score more than public leaderboard

---

## 📈 Expected Cumulative Improvement

If you do all 4 quick improvements:
- **Current:** RMSE 0.112898
- **Target:** RMSE 0.105-0.110
- **Improvement:** ~0.003-0.008 RMSE

This could move you from **0.62673** to **~0.115-0.120** on Kaggle leaderboard!

---

## 🆘 Need Help?

Check `docs/IMPROVEMENT_ROADMAP.md` for detailed explanations of each technique.

Good luck! 🚀

