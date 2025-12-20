# Breaking Through the RMSE Plateau - Advanced Strategy

**Current Best:** CatBoost RMSLE 0.12973 (CV RMSE 0.12017)  
**Target:** RMSLE < 0.125 (CV RMSE < 0.115)  
**Status:** Stuck at plateau - need advanced techniques

---

## 🔍 Diagnosis: Why You're Stuck

### Current Situation Analysis

1. **CV-Kaggle Gap is Good (0.00956)** ✅
   - This means you're NOT overfitting
   - The model generalizes well
   - **Problem:** You've hit the model's capacity limit

2. **Ensemble Models Show Promise But Fail** ⚠️
   - Stacking CV: 0.11184 (excellent!)
   - Blending CV: 0.11194 (excellent!)
   - **Problem:** Numerical instability prevents deployment

3. **Feature Engineering is Good But Not Optimal** ⚠️
   - 248-251 features after selection
   - Target encoding implemented
   - **Problem:** May be missing key interactions or domain features

4. **Hyperparameter Search is Limited** ⚠️
   - Optuna with 20-40 trials
   - **Problem:** May not be exploring full search space

---

## 🚀 Advanced Techniques to Break Through

### Priority 1: Fix Ensemble Models (HIGHEST IMPACT) ⭐⭐⭐

**Expected Improvement:** 0.005-0.015 RMSLE (from 0.11184 CV → ~0.120-0.125 Kaggle)

#### 1.1 Fix Stacking Numerical Instability

**Root Cause:** Predictions in wrong space or unbounded values

**Solution:**
```python
# In 11stackingModel.py, before expm1():
final_pred_log = meta_model.predict(oof_test)

# ADD BOUNDS CHECKING:
# Clip to reasonable log space bounds
log_min = np.log1p(10000)  # $10k minimum
log_max = np.log1p(2000000)  # $2M maximum
final_pred_log = np.clip(final_pred_log, log_min, log_max)

# Then transform
final_pred_real = np.expm1(final_pred_log)

# Double-check bounds in real space
final_pred_real = np.clip(final_pred_real, 10000, 2000000)
```

**Action:** Update `notebooks/Models/11stackingModel.py` with bounds checking

#### 1.2 Try XGBoost/Ridge as Meta-Model

**Why:** Lasso (α=0.0005) may be too weak, causing instability

**Solution:**
```python
# Option 1: Ridge meta-model (more stable)
meta_model = Ridge(alpha=0.1)

# Option 2: XGBoost meta-model (non-linear)
meta_model = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)
```

**Action:** Update `config_local/model_config.py` STACKING config

#### 1.3 Fix Blending Space Consistency

**Problem:** Blending mixes predictions from different spaces

**Solution:**
```python
# Ensure all predictions are in REAL space before blending
# Load predictions, convert to real space if needed
for name, pred_df in predictions.items():
    if pred_df['SalePrice'].mean() < 100:  # Likely in log space
        pred_df['SalePrice'] = np.expm1(pred_df['SalePrice'])
    
# Then blend in real space
blend['SalePrice'] = weighted_average_in_real_space
```

**Action:** Update `notebooks/Models/10blendingModel.py`

---

### Priority 2: Advanced Feature Engineering ⭐⭐⭐

**Expected Improvement:** 0.002-0.008 RMSLE

#### 2.1 Neighborhood-Based Features

**Why:** Location is critical for house prices

**Features to Add:**
```python
# Neighborhood price statistics (already in stage 8, but enhance)
- Neighborhood_median_price
- Neighborhood_price_per_sqft
- Neighborhood_age_distribution
- Neighborhood_quality_distribution

# Distance-based features (if coordinates available)
- Distance_to_center
- Distance_to_schools
- Distance_to_parks
```

**Action:** Enhance `notebooks/preprocessing/8targetEncoding.py`

#### 2.2 Advanced Interaction Features

**Why:** Current interactions may miss key relationships

**Features to Add:**
```python
# Quality × Location interactions
- OverallQual × Neighborhood_mean_logSP
- KitchenQual × Neighborhood_mean_logSP

# Size × Quality × Location
- TotalSF × OverallQual × Neighborhood_mean_logSP

# Age × Condition × Location
- Age × OverallCond × Neighborhood_mean_logSP
```

**Action:** Add to `notebooks/preprocessing/4featureEngineering.py`

#### 2.3 Polynomial Features (Higher Order)

**Why:** Current only uses squared terms

**Features to Add:**
```python
# Cubic terms for key features
- TotalSF_cubed
- OverallQual_cubed
- GrLivArea_cubed

# Cross-polynomials
- TotalSF × OverallQual²
- GrLivArea × Age²
```

**Action:** Add to `notebooks/preprocessing/4featureEngineering.py`

---

### Priority 3: Pseudo-Labeling ⭐⭐

**Expected Improvement:** 0.003-0.010 RMSLE  
**Risk:** Medium (can cause overfitting if not careful)

#### 3.1 Implementation Strategy

**Step 1:** Train model on training data
**Step 2:** Predict on test set
**Step 3:** Select high-confidence predictions (close to training distribution)
**Step 4:** Add to training set and retrain

**Code:**
```python
# 1. Train initial model
model.fit(X_train, y_train)

# 2. Predict on test
test_pred = model.predict(X_test)
test_pred_real = np.expm1(test_pred)

# 3. Select confident predictions (within 2 std of training mean)
train_mean = np.expm1(y_train.mean())
train_std = np.expm1(y_train.std())

mask = (test_pred_real >= train_mean - 2*train_std) & \
       (test_pred_real <= train_mean + 2*train_std)

# 4. Create pseudo-labeled data
X_pseudo = X_test[mask]
y_pseudo = test_pred[mask]  # In log space

# 5. Combine and retrain
X_combined = np.vstack([X_train, X_pseudo])
y_combined = np.hstack([y_train, y_pseudo])

model.fit(X_combined, y_combined)
```

**Action:** Create `notebooks/Models/12pseudoLabeling.py`

---

### Priority 4: More Aggressive Hyperparameter Search ⭐⭐

**Expected Improvement:** 0.001-0.003 RMSSE

#### 4.1 Increase Optuna Trials

**Current:** 20-40 trials  
**Recommended:** 100-200 trials for best models

**Action:** Update `config_local/model_config.py`:
```python
CATBOOST = {
    "optuna_settings": {
        "n_trials": 100,  # Increase from 20-30
        "n_splits": 5,
        "random_state": 42,
    }
}
```

#### 4.2 Wider Search Space

**Current:** Limited ranges  
**Recommended:** Explore more extreme values

**Action:** Expand search spaces in `config_local/model_config.py`

---

### Priority 5: Advanced Cross-Validation ⭐⭐

**Expected Improvement:** 0.001-0.003 RMSLE

#### 5.1 GroupKFold by Neighborhood

**Why:** Prevents data leakage from similar neighborhoods

**Implementation:**
```python
from sklearn.model_selection import GroupKFold

# Group by Neighborhood
groups = train['Neighborhood'].values
gkf = GroupKFold(n_splits=5)

for train_idx, val_idx in gkf.split(X, y, groups):
    # Train/validate
```

**Action:** Update model training scripts

#### 5.2 StratifiedKFold by Price Bins

**Why:** Ensures balanced price distribution in folds

**Implementation:**
```python
from sklearn.model_selection import StratifiedKFold

# Create price bins
price_bins = pd.cut(y, bins=5, labels=False)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in skf.split(X, y, price_bins):
    # Train/validate
```

**Action:** Update model training scripts

---

### Priority 6: Neural Networks (Optional) ⭐

**Expected Improvement:** 0.002-0.005 RMSLE  
**Complexity:** High

#### 6.1 TabNet or Neural Network

**Why:** Can capture complex non-linear patterns

**Options:**
- TabNet (PyTorch)
- Simple MLP (scikit-learn)
- LightGBM + Neural Network ensemble

**Action:** Create `notebooks/Models/13neuralNetwork.py`

---

## 📊 Implementation Roadmap

### Week 1: Fix Ensembles (Highest Priority)
1. ✅ Fix stacking bounds checking
2. ✅ Try Ridge/XGBoost meta-model
3. ✅ Fix blending space consistency
4. ✅ Test and submit

**Expected:** 0.005-0.015 improvement

### Week 2: Advanced Features
1. ✅ Add neighborhood-based features
2. ✅ Add advanced interactions
3. ✅ Add higher-order polynomials
4. ✅ Retrain models

**Expected:** 0.002-0.008 improvement

### Week 3: Pseudo-Labeling
1. ✅ Implement pseudo-labeling
2. ✅ Test carefully (monitor overfitting)
3. ✅ Retrain best models

**Expected:** 0.003-0.010 improvement

### Week 4: Optimization
1. ✅ Increase Optuna trials
2. ✅ Implement GroupKFold
3. ✅ Final ensemble optimization

**Expected:** 0.001-0.003 improvement

---

## 🎯 Realistic Targets

### Conservative (Do Priority 1-2)
- **Current:** 0.12973
- **Target:** 0.125-0.127
- **Improvement:** 0.002-0.005

### Moderate (Do Priority 1-3)
- **Current:** 0.12973
- **Target:** 0.120-0.125
- **Improvement:** 0.005-0.010

### Aggressive (Do All)
- **Current:** 0.12973
- **Target:** 0.115-0.120
- **Improvement:** 0.010-0.015

---

## ⚠️ Important Notes

1. **Monitor Overfitting:** Watch CV-Kaggle gap (should stay <0.02)
2. **Test Incrementally:** Don't implement everything at once
3. **Save Checkpoints:** Keep predictions from each experiment
4. **Focus on Ensembles First:** Highest impact, lowest risk

---

## 🔧 Quick Start Commands

```bash
# 1. Fix stacking (highest priority)
# Edit notebooks/Models/11stackingModel.py
# Add bounds checking before expm1()

# 2. Test stacking fix
python notebooks/Models/11stackingModel.py

# 3. Submit and check score
python scripts/submit_model.py stacking

# 4. If successful, proceed to blending fix
python notebooks/Models/10blendingModel.py
```

---

*Last Updated: December 2025*  
*Focus: Breaking through 0.12973 RMSLE plateau*

