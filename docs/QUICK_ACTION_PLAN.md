# Quick Action Plan - Improve Leaderboard NOW

**Current Best:** Stacking RMSE 0.112898 (Kaggle: 0.62673)

---

## 🚀 DO THESE 3 THINGS FIRST (Highest Impact!)

### 1. Retrain Models with Process8 ⭐⭐⭐
**Impact:** 0.002-0.010 RMSE improvement
**Time:** 1-2 hours

```bash
python notebooks/Models/7XGBoostModel.py
python notebooks/Models/8lightGbmModel.py
python notebooks/Models/9catBoostModel.py
python notebooks/Models/11stackingModel.py
```

**Why:** Models haven't been retrained with new features yet!

---

### 2. Try XGBoost Meta-Model ⭐⭐⭐
**Impact:** 0.003-0.008 RMSE improvement
**Time:** 30 minutes

**Step 1:** Update `notebooks/Models/11stackingModel.py` around line 140:

```python
# Change from:
if cfg["meta_model"] == "lasso":
    meta_model = Lasso(**cfg["meta_model_params"])
elif cfg["meta_model"] == "ridge":
    meta_model = Ridge(**cfg["meta_model_params"])
else:
    meta_model = Ridge() # Default

# To:
if cfg["meta_model"] == "lasso":
    meta_model = Lasso(**cfg["meta_model_params"])
elif cfg["meta_model"] == "ridge":
    meta_model = Ridge(**cfg["meta_model_params"])
elif cfg["meta_model"] == "xgboost":
    meta_model = XGBRegressor(**cfg["meta_model_params"])
else:
    meta_model = Ridge() # Default
```

**Step 2:** Update `config_local/model_config.py` STACKING section:

```python
STACKING = {
    ...
    "meta_model": "xgboost",  # Change from "lasso"
    "meta_model_params": {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.05,
        "random_state": 42,
        "device": "cuda"  # If you have GPU
    },
    ...
}
```

**Step 3:** Retrain stacking:
```bash
python notebooks/Models/11stackingModel.py
```

---

### 3. Run Optimized Blending ⭐⭐
**Impact:** 0.001-0.005 RMSE improvement
**Time:** 5 minutes

```bash
python notebooks/Models/10blendingModel.py
```

**Why:** Automatically finds best weights!

---

## 📊 Expected Results

After doing all 3:
- **Current:** RMSE 0.112898 (Kaggle: 0.62673)
- **Expected:** RMSE 0.105-0.110 (Kaggle: 0.115-0.120)
- **Improvement:** ~0.003-0.008 RMSE

---

## ✅ Checklist

- [ ] Retrain XGBoost with process8
- [ ] Retrain LightGBM with process8
- [ ] Retrain CatBoost with process8
- [ ] Retrain Stacking with process8
- [ ] Update stacking to support XGBoost meta-model
- [ ] Change config to use XGBoost meta-model
- [ ] Retrain stacking with XGBoost meta-model
- [ ] Run optimized blending
- [ ] Compare results
- [ ] Submit best model to Kaggle

---

## 🎯 Next Steps After These

See `docs/LEADERBOARD_IMPROVEMENT_PLAN.md` for:
- Ridge meta-model
- Multi-level stacking
- Pseudo-labeling
- Feature selection
- And more!

---

**Start now - these 3 improvements will give you the biggest boost!** 🚀

