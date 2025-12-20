# Ensemble and CV Strategy Fixes

**Date**: 2025-12-20  
**Status**: Implemented

## Summary

This document describes the fixes implemented to address the four priority items from the TODO list:
1. Fix ensemble space consistency
2. Fix CV strategy
3. Increase model diversity
4. Error-driven feature engineering

---

## 1. Fix Ensemble Space Consistency ✅

### Problem
- Blending was mixing log-space and real-space predictions
- Some models output log-space, others real-space
- This caused numerical explosions (1e17/1e60 values)

### Solution
- **Blending Model (`10blendingModel.py`)**:
  - Added `ensure_log_space()` function to detect and convert predictions
  - Modified blending to work entirely in log space
  - When OOF test predictions are available, use them directly (already in log space)
  - When blending from CSV files, convert to log space first, blend, then convert back
  - All blending operations now consistent in log space

- **Stacking Model (`11stackingModel.py`)**:
  - Already correct (all base models predict in log space, meta-model works in log space)
  - Final conversion to real space happens once at the end

### Files Modified
- `notebooks/Models/10blendingModel.py`
- `notebooks/Models/11stackingModel.py` (verified correct)

---

## 2. Fix CV Strategy ✅

### Problem
- Standard KFold was mixing cheap/expensive houses across folds
- CV scores didn't reflect Kaggle test distribution
- Ridge CV RMSE ≈ 0.096 but Kaggle RMSLE ≈ 1.41 (huge gap)

### Solution
- **Created `utils/cv_strategy.py`**:
  - `create_stratified_cv_splits()`: Creates stratified CV based on target quantiles
  - Bins target values into quantiles (default 10 bins)
  - Uses `StratifiedKFold` to ensure each fold has similar target distribution
  - Better reflects Kaggle test distribution

- **Updated Models**:
  - `11stackingModel.py`: Now uses stratified CV
  - `utils/optimization.py`: Added `cv_strategy` parameter (defaults to "stratified")
  - All Optuna studies now use stratified CV by default

### Files Created/Modified
- `utils/cv_strategy.py` (new)
- `notebooks/Models/11stackingModel.py`
- `utils/optimization.py`

### Usage
```python
from utils.cv_strategy import get_cv_strategy

# Stratified CV (recommended)
cv_splits = get_cv_strategy(
    strategy="stratified",
    y=y_target,
    n_splits=5,
    random_state=42
)

# Standard KFold (if needed)
kf = get_cv_strategy(strategy="kfold", n_splits=5, shuffle=True, random_state=42)
```

---

## 3. Increase Model Diversity ✅

### Problem
- Models too correlated (>0.95 correlation)
- Ridge dominating ensembles despite poor Kaggle performance
- Lack of diversity in feature views

### Solution
- **Removed Ridge from Ensembles**:
  - Removed from `BLENDING` config
  - Removed from `STACKING` base_models
  - Updated `10blendingModel.py` to remove Ridge from mapping
  - Reason: Ridge has CV-Kaggle gap >1.0 (severe overfitting)

- **Created Error Analysis Script**:
  - `scripts/analyze_model_errors.py`: Analyzes prediction errors
  - Identifies worst predictions
  - Groups errors by Neighborhood, OverallQual, YearBuilt, etc.
  - Suggests targeted features based on error patterns

### Files Modified
- `config_local/model_config.py` (removed Ridge from ensembles)
- `notebooks/Models/10blendingModel.py` (removed Ridge mapping)

### Files Created
- `scripts/analyze_model_errors.py`

### Next Steps (To Do)
- Train models on different feature sets:
  - `process6` (raw one-hot encoding)
  - `process8` (target-encoded) - current
  - Future: `cb_raw` (simplified for CatBoost)
- Try different loss functions for CatBoost:
  - `loss_function="MAE"`
  - `loss_function="Quantile:alpha=0.9"`

---

## 4. Error-Driven Feature Engineering ✅

### Solution
- **Created `scripts/analyze_model_errors.py`**:
  - Loads OOF predictions from best model
  - Calculates errors in both log and real space
  - Analyzes worst predictions (top 5%)
  - Groups errors by key features:
    - Neighborhood
    - OverallQual
    - YearBuilt
    - YearRemodAdd
    - GrLivArea, TotalSF, etc.
  - Suggests new features based on patterns:
    - `Is_NewHouse = YearBuilt > 2000`
    - `Qual_Age_Interaction = OverallQual * (2024 - YearBuilt)`
    - `RemodAge = YearRemodAdd - YearBuilt`
    - `Is_Remodeled = (YearRemodAdd != YearBuilt)`
    - Neighborhood-specific adjustments

### Usage
```bash
python scripts/analyze_model_errors.py catboost
```

### Output
- Saves analysis to `runs/error_analysis/`
- CSV files with worst predictions and feature-grouped errors
- Console output with suggested features

---

## Testing Recommendations

1. **Test Ensemble Fixes**:
   ```bash
   # Run stacking to generate OOF predictions
   python notebooks/Models/11stackingModel.py
   
   # Run blending (should now work correctly in log space)
   python notebooks/Models/10blendingModel.py
   ```

2. **Test CV Strategy**:
   - Retrain models with new stratified CV
   - Compare CV scores: should be slightly higher but more realistic
   - Compare Kaggle scores: should improve (better generalization)

3. **Test Error Analysis**:
   ```bash
   python scripts/analyze_model_errors.py catboost
   ```
   - Review suggested features
   - Implement top suggestions
   - Retrain and validate

---

## Expected Impact

1. **Ensemble Space Consistency**:
   - No more numerical explosions
   - Predictions in reasonable range ($10k-$2M)
   - Better ensemble performance

2. **CV Strategy**:
   - CV scores more realistic (slightly higher but closer to Kaggle)
   - Better model selection
   - Improved generalization

3. **Model Diversity**:
   - Removing Ridge should improve ensemble performance
   - Error analysis will guide targeted feature engineering
   - Future: Different feature sets and loss functions will increase diversity

4. **Error-Driven Features**:
   - 3-5 targeted features > 50 generic ones
   - Addresses specific failure patterns
   - Should help break through 0.125 RMSLE barrier

---

## Notes

- All changes are backward compatible
- Existing OOF predictions will be regenerated with new CV strategy
- Stratified CV is now the default for all new model training
- Ridge can still be trained individually, just not in ensembles

---

## Next Steps

1. ✅ Fix ensemble space consistency - DONE
2. ✅ Fix CV strategy - DONE
3. ✅ Remove Ridge from ensembles - DONE
4. ✅ Create error analysis script - DONE
5. ⏳ Run error analysis on best model
6. ⏳ Implement suggested features from error analysis
7. ⏳ Train models on different feature sets (process6, process8)
8. ⏳ Try different CatBoost loss functions
9. ⏳ Retrain ensembles with new diverse models
10. ⏳ Validate improvements on Kaggle

