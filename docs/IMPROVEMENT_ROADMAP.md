# Model Improvement Roadmap & Action Plan

**Last Updated: December 2025**

This document consolidates all improvement strategies, action plans, and quick wins to improve model performance and leaderboard ranking.

**Current Best Performance:**
- CatBoost: RMSE 0.12017 (Kaggle: 0.12973) ⭐ Best (as of 2025-12-19)
- XGBoost: RMSE 0.11436 (Kaggle: 0.13094, latest: 2025-12-20)
- LightGBM: RMSE 0.11795
- Stacking: RMSE 0.11184 (Kaggle: 3.18379) ⚠️ Exploded predictions

**Current Status:** 248-251 features (process8), 8-stage preprocessing, 12 models

---

## 🚀 Priority 1: IMMEDIATE WINS (Do These First!)

### 1. Retrain Models with Process8 Data ⭐⭐⭐
**Expected Impact:** 0.002-0.010 RMSE improvement  
**Time:** 1-2 hours  
**Status:** ✅ Process8 data available

**Why:** Models should use the latest preprocessing with target encoding and feature selection.

**Action:**
```bash
# Retrain all models with process8 data
python notebooks/Models/7XGBoostModel.py
python notebooks/Models/8lightGbmModel.py
python notebooks/Models/9catBoostModel.py
python notebooks/Models/11stackingModel.py
```

**Expected:** Lower CV scores, better Kaggle performance

---

### 2. Fix Blending and Stacking Models ⭐⭐⭐
**Expected Impact:** Critical fix (currently producing exploded predictions)  
**Time:** 1-2 hours

**Issues:**
- Blending: Mean prediction ~1.5e17 (should be ~$178k)
- Stacking: Mean prediction ~1.5e60 (should be ~$178k)

**Root Causes:**
- Space mismatch (log vs. real space)
- Numerical instability in meta-model
- Missing bounds checking

**Fixes Needed:**
1. Ensure consistent space (all predictions in real space for blending)
2. Add bounds checking before `expm1()` in stacking
3. Increase Lasso alpha or use Ridge for meta-model
4. Clip final predictions to reasonable range ($10k-$2M)

**Action:** See `docs/CONSOLIDATION_AND_ENHANCEMENT.md` for details

---

### 3. Try XGBoost as Stacking Meta-Model ⭐⭐⭐
**Expected Impact:** 0.003-0.008 RMSE improvement  
**Time:** 30 minutes

**Why:** XGBoost often outperforms Lasso as meta-learner for stacking.

**Action:** Update `config_local/model_config.py`:
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

Then update `notebooks/Models/11stackingModel.py` to support XGBoost meta-model.

---

### 4. Run Optimized Blending ⭐⭐
**Expected Impact:** 0.001-0.005 RMSE improvement  
**Time:** 5 minutes

**Why:** Automatic weight optimization finds better ensemble weights.

**Action:**
```bash
python notebooks/Models/10blendingModel.py
```

**Note:** Fix blending model first (see Priority 1, Item 2)

---

## 🎯 Priority 2: HIGH-IMPACT IMPROVEMENTS (This Week)

### 5. Try Ridge as Stacking Meta-Model ⭐⭐
**Expected Impact:** 0.002-0.005 RMSE improvement  
**Time:** 15 minutes

**Why:** Ridge often works better than Lasso for stacking (less aggressive regularization).

**Action:** Update config:
```python
"meta_model": "ridge",
"meta_model_params": {
    "alpha": 0.1,  # Try: 0.01, 0.1, 1.0, 10.0
    "random_state": 42,
}
```

---

### 6. Increase Optuna Trials ⭐⭐
**Expected Impact:** 0.001-0.003 RMSE improvement  
**Time:** +30-60 min per model

**Why:** More trials = better hyperparameters.

**Action:** Update `config_local/model_config.py`:
```python
XGBOOST = {
    ...
    "optuna_settings": {
        "n_trials": 200,  # Increase from current
        ...
    }
}
```

---

### 7. Better Cross-Validation Strategy ⭐⭐
**Expected Impact:** 0.002-0.005 RMSE improvement  
**Time:** 1-2 hours

**Why:** GroupKFold by Neighborhood prevents leakage.

**Action:** Modify stacking to use GroupKFold:
```python
from sklearn.model_selection import GroupKFold

# Group by Neighborhood to prevent leakage
groups = train["Neighborhood"].values
gkf = GroupKFold(n_splits=5)
for train_idx, val_idx in gkf.split(X, y, groups):
    ...
```

---

## 🔬 Priority 3: ADVANCED TECHNIQUES (Next Week)

### 8. Multi-Level Stacking ⭐⭐⭐
**Expected Impact:** 0.005-0.015 RMSE improvement  
**Time:** 3-5 hours

**Concept:**
- Level 1: Base models (XGBoost, LightGBM, CatBoost)
- Level 2: Meta-model 1 (combines Level 1)
- Level 3: Meta-model 2 (combines Level 2 + original features)

**Implementation:** Create `notebooks/Models/12multiLevelStacking.py`

---

### 9. Pseudo-Labeling ⭐⭐
**Expected Impact:** 0.003-0.010 RMSE improvement  
**Time:** 2-3 hours

**Concept:** Use confident test predictions to augment training data.

**Steps:**
1. Train model on training data
2. Predict on test set
3. Select high-confidence predictions (close to training distribution)
4. Add to training set and retrain

**Risk:** Can cause overfitting if not done carefully

---

### 10. Feature Selection with SHAP ⭐⭐
**Expected Impact:** 0.002-0.008 RMSE improvement  
**Time:** 2-3 hours

**Why:** Remove noisy features, keep only important ones.

**Action:** Feature selection already implemented in stage 7, but can be enhanced with SHAP values.

---

## 📊 Expected Cumulative Improvement

### If You Do Priority 1 (Immediate):
- **Current:** RMSE 0.12017 (CatBoost, Kaggle: 0.12973 - best as of 2025-12-19)
- **Target:** RMSE 0.110-0.115 (Kaggle: 0.120-0.125)
- **Improvement:** ~0.005-0.010 RMSE

### If You Do Priority 1-2 (This Week):
- **Target:** RMSE 0.105-0.110 (Kaggle: 0.115-0.120)
- **Improvement:** ~0.010-0.015 RMSE

### If You Do Priority 1-3 (Full Plan):
- **Target:** RMSE 0.100-0.105 (Kaggle: 0.110-0.115)
- **Improvement:** ~0.015-0.020 RMSE

---

## 🎯 Recommended Order

### **Today (2-3 hours):**
1. ✅ Fix blending and stacking models (CRITICAL)
2. ✅ Retrain all models with process8
3. ✅ Try XGBoost meta-model for stacking
4. ✅ Run optimized blending (after fix)
5. ✅ Compare results

### **This Week (5-10 hours):**
6. ✅ Try Ridge meta-model
7. ✅ Increase Optuna trials
8. ✅ Implement GroupKFold CV
9. ✅ Compare all results

### **Next Week (10-15 hours):**
10. ✅ Multi-level stacking
11. ✅ Pseudo-labeling (carefully)
12. ✅ Enhanced feature selection
13. ✅ Final ensemble optimization

---

## 🔍 Diagnostic Steps (Do First!)

Before implementing new features, diagnose current issues:

### 1. Error Analysis
```python
# Analyze where models fail
# - Which neighborhoods are hardest to predict?
# - Which price ranges have highest error?
# - Are there systematic biases?
```

### 2. Feature Importance Analysis
```python
# Use SHAP to understand feature importance
# - Which features are most important?
# - Are there redundant features?
# - Missing important interactions?
```

### 3. Overfitting Check
```python
# Compare CV score vs Kaggle score
# - Large gap = overfitting
# - Small gap = underfitting or data mismatch
cv_score = 0.120
kaggle_score = 0.130
gap = kaggle_score - cv_score  # Should be < 0.02
```

---

## 💡 Pro Tips

1. **Start Small:** Don't implement everything at once. Test each improvement individually.

2. **Version Control:** Save predictions from each experiment to compare.

3. **Ensemble Diversity:** Make sure your base models are diverse (different algorithms, different hyperparameters).

4. **Validation Strategy:** Use a holdout set or GroupKFold to prevent overfitting.

5. **Kaggle Leaderboard:** Don't overfit to public leaderboard. Focus on CV score.

6. **Monitor CV Gap:** If gap > 0.02, you're overfitting.

7. **Trust CV Score:** More reliable than public leaderboard.

---

## 🐛 Troubleshooting

### Issue: Models don't improve
- **Check:** Are new features being used?
- **Fix:** Verify model scripts use process8 data
- **Check:** Feature importance - are new features used?

### Issue: Overfitting (CV gap > 0.02)
- **Fix:** Use more regularization
- **Fix:** Consider feature selection
- **Fix:** Reduce number of features

### Issue: Blending/Stacking exploded
- **Fix:** Check space consistency (log vs. real)
- **Fix:** Add bounds checking
- **Fix:** Use Ridge instead of Lasso for meta-model

---

## 📈 Success Metrics

### What Success Looks Like:
✅ **CV RMSE improves** by 0.002-0.010  
✅ **Kaggle score improves** by 0.01-0.02  
✅ **CV gap stays small** (< 0.02)  
✅ **New features are important** (check SHAP values)

### If Not Improving:
⚠️ **Check feature importance** - Are new features being used?  
⚠️ **Check correlations** - Are features redundant?  
⚠️ **Check for leakage** - Any suspicious correlations?  
⚠️ **Try feature selection** - Remove noise

---

## 🎯 Quick Start Commands

```bash
# 1. Fix and retrain models with process8
python notebooks/Models/7XGBoostModel.py
python notebooks/Models/8lightGbmModel.py
python notebooks/Models/9catBoostModel.py

# 2. Fix stacking model (update meta-model)
python notebooks/Models/11stackingModel.py

# 3. Fix and run blending
python notebooks/Models/10blendingModel.py

# 4. Compare results
python scripts/quick_model_comparison.py
python scripts/compare_models.py

# 5. Submit best model
python scripts/submit_model.py catboost
```

---

## 📚 Related Documentation

- **Preprocessing:** `docs/PREPROCESSING_REFACTORING.md`
- **Data Leakage:** `docs/DATA_LEAKAGE_ANALYSIS.md`
- **Feature Engineering:** `docs/ADVANCED_FEATURES_IMPLEMENTED.md`
- **Feature Selection:** `docs/FEATURE_SELECTION_GUIDE.md`
- **Target Encoding:** `docs/TARGET_ENCODING_EXPLAINED.md`

---

**Start with Priority 1 - these are the highest-impact improvements!** 🚀

*Focus on fixing blending/stacking first, then retrain with process8 data.*
