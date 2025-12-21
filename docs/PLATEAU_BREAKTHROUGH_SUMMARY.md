# Plateau Breakthrough - Action Summary

**Date:** December 2025  
**Status:** In Progress  
**Current Best:** CatBoost RMSLE 0.12973 (CV RMSE 0.12017)

---

## ✅ Completed Actions

### 1. Fixed Ensemble Models (Numerical Stability)

**Stacking Model:**
- ✅ Added bounds checking before `expm1()` transformation
- ✅ Changed meta-model from Lasso (α=0.0005) → Ridge (α=0.1) for stability
- ✅ Added clipping in both log space (10.7-14.5) and real space ($10k-$2M)
- ✅ **Result:** Predictions now reasonable ($179,647 mean) but Kaggle score: 0.13478 (worse than best)

**Blending Model:**
- ✅ Added space consistency checks (ensure all predictions in real space)
- ✅ Added bounds checking after blending
- ✅ **Result:** Predictions reasonable ($179,714 mean) but Kaggle score: 0.13410 (worse than best)

**Analysis:**
- CV scores excellent (0.1118-0.1119) but don't translate to Kaggle
- Suggests overfitting to CV folds or insufficient model diversity
- **Decision:** Focus on improving single best model instead

### 2. Increased CatBoost Optimization

**Changes Made:**
- ✅ Increased Optuna trials: 10 → 100
- ✅ Increased CV folds: 3 → 5
- ✅ Expanded search space:
  - iterations: (500-1200) → (300-2000)
  - learning_rate: (0.02-0.08) → (0.01-0.1)
  - depth: (4-7) → (4-10)
  - l2_leaf_reg: (1-8) → (1-10)
  - bagging_temperature: (0-1) → (0-2)
  - random_strength: (0-1) → (0-2)

**Expected Impact:** 0.001-0.003 RMSLE improvement

---

## 🔄 Next Steps (Recommended Order)

### Priority 1: Retrain CatBoost with Enhanced Optimization

**Action:**
```bash
python notebooks/Models/9catBoostModel.py
```

**Expected:** Better hyperparameters found with 100 trials  
**Time:** ~2-3 hours (100 trials × 5-fold CV)

### Priority 2: Add Neighborhood × Quality Interactions

**Why:** Location × Quality is a powerful predictor (premium neighborhoods with high quality = very expensive)

**Implementation:** Add to `notebooks/preprocessing/8targetEncoding.py` after target encoding:
```python
# After target encoding, add interactions
if "Neighborhood_mean_logSP" in train.columns:
    # Quality × Location interactions
    train["Qual_x_Neighborhood"] = train["OverallQual"] * train["Neighborhood_mean_logSP"]
    test["Qual_x_Neighborhood"] = test["OverallQual"] * test["Neighborhood_mean_logSP"]
    
    # Size × Location interactions
    train["TotalSF_x_Neighborhood"] = train["TotalSF"] * train["Neighborhood_mean_logSP"]
    test["TotalSF_x_Neighborhood"] = test["TotalSF"] * test["Neighborhood_mean_logSP"]
```

**Expected Impact:** 0.002-0.005 RMSLE improvement

### Priority 3: Implement Pseudo-Labeling

**Why:** Use confident test predictions to augment training data

**Implementation:** Create `notebooks/Models/12pseudoLabeling.py`

**Expected Impact:** 0.003-0.010 RMSLE improvement  
**Risk:** Medium (can cause overfitting if not careful)

### Priority 4: GroupKFold by Neighborhood

**Why:** Prevents data leakage from similar neighborhoods in CV

**Expected Impact:** 0.001-0.003 RMSLE improvement

---

## 📊 Current Situation Analysis

### Why Ensembles Failed

1. **Base Models Too Similar:**
   - XGBoost, LightGBM, CatBoost: >0.95 correlation
   - Low diversity = limited ensemble benefit

2. **CV Overfitting:**
   - Excellent CV scores (0.1118) but worse Kaggle (0.13478)
   - Gap: 0.023 (suggests overfitting to CV structure)

3. **Meta-Model Issues:**
   - Ridge meta-model may be too simple
   - Need more diverse base models or better meta-model

### Why Single Model (CatBoost) is Best

1. **Best Generalization:**
   - CV-Kaggle gap: 0.00956 (excellent, minimal overfitting)
   - Better than ensembles despite lower CV score

2. **Optimal Hyperparameters:**
   - Already well-tuned (depth=5, lr=0.048, iterations=352)
   - But can be improved with more trials

3. **Categorical Handling:**
   - Native support for categoricals
   - Better than one-hot encoding for high-cardinality features

---

## 🎯 Realistic Targets

### Conservative (Retrain CatBoost + Add Features)
- **Current:** 0.12973
- **Target:** 0.127-0.128
- **Improvement:** 0.001-0.003

### Moderate (Above + Pseudo-Labeling)
- **Current:** 0.12973
- **Target:** 0.120-0.125
- **Improvement:** 0.005-0.010

### Aggressive (All Techniques)
- **Current:** 0.12973
- **Target:** 0.115-0.120
- **Improvement:** 0.010-0.015

---

## ⚠️ Key Learnings

1. **Ensembles aren't always better** - Single best model can outperform ensembles if base models are too similar
2. **CV score ≠ Kaggle score** - Need to watch CV-Kaggle gap (should be <0.02)
3. **Model diversity matters** - Need diverse base models for ensemble benefits
4. **Bounds checking critical** - Numerical stability issues can cause catastrophic failures

---

## 🔧 Quick Commands

```bash
# 1. Retrain CatBoost with enhanced optimization (2-3 hours)
python notebooks/Models/9catBoostModel.py

# 2. Add neighborhood interactions (edit 8targetEncoding.py, then rerun preprocessing)
python notebooks/preprocessing/run_preprocessing.py

# 3. Retrain models with new features
python notebooks/Models/9catBoostModel.py

# 4. Submit best model
python scripts/submit_model.py catboost
```

---

*Focus: Improve single best model (CatBoost) rather than ensembles*

