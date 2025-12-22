# Project Status & Progress

**Last Updated:** December 2025  
**Current Best Score:** RMSLE 0.12973 (CatBoost, CV RMSE 0.12017)

---

## Current Status

### ✅ Recently Completed

1. **Fixed Ensemble Models** ✅
   - **Stacking:** Added bounds checking, changed to Ridge meta-model
     - Predictions: Fixed (no explosion) ✅
     - Score: 0.13478 (worse than best, but stable)
   - **Blending:** Added bounds checking and space consistency
     - Predictions: Fixed (no explosion) ✅
     - Score: 0.13410 (worse than best, but stable)
   - **Key Finding:** Ensembles show excellent CV (0.1118) but worse Kaggle scores. Base models too similar (>0.95 correlation).

2. **Enhanced CatBoost Optimization** ✅
   - Increased Optuna trials: 10 → **100**
   - Increased CV folds: 3 → **5**
   - Expanded search space (wider ranges)

3. **Feature Engineering** ✅
   - Neighborhood interactions already implemented in stage 8
   - Advanced interactions already in place
   - Error-driven features implemented: Qual_Age_Interaction, RemodAge_FromBuild, Is_Remodeled, OverallQual_Squared

4. **CV Strategy Improvements** ✅
   - Implemented stratified CV strategy
   - Created `utils/cv_strategy.py`
   - Removed Ridge from ensembles

---

## Key Insights

1. **Single model > Ensemble** (when base models are similar)
2. **CV score ≠ Kaggle score** (watch the gap - current gap is small at 0.00956)
3. **Bounds checking critical** (prevented catastrophic failures)
4. **More optimization helps** (100 trials > 10 trials)
5. **Simpler is better for CatBoost** - CatBoost works best with raw categoricals and simple numerics, not over-engineered features

---

## Analysis: Why Progress Has Plateaued

### Good News ✅
1. **CV-Kaggle gap is small (0.00956)** - Not overfitting
2. **Model generalizes well** - Good foundation
3. **Feature engineering is solid** - 248-251 features

### The Problem ⚠️
1. **Hit model capacity limit** - Current features + hyperparameters maxed out
2. **Ensembles don't help** - Base models too similar
3. **Need more data or better features** - Limited by dataset

---

## Recommended Next Steps

### Immediate (High Priority)
1. **Simplify CatBoost Pipeline** - Create `process_cb_raw.csv` with raw categoricals, drop one-hot encoding, target encoding, scaling
2. **Retrain CatBoost** with new features (254 features, includes error-driven features)
3. **Analyze errors** - Which houses/neighborhoods are hardest to predict?

### Short-term (This Week)
1. **GroupKFold by Neighborhood** - Better CV strategy
2. **Model diversity improvements** - Train on different feature sets (process6, process8, cb_raw)
3. **Different loss functions** - Try CatBoost with `loss_function="MAE"` or `"Quantile:alpha=0.9"`

### Advanced (After Pipeline Stable)
1. **Pseudo-labeling** - Use confident test predictions to augment training
2. **Neural networks** - TabNet or simple MLP
3. **External data** - Add location data, market trends

---

## Realistic Expectations

- **CatBoost (100 trials):** 0.127-0.129 (0.001-0.003 improvement)
- **With pseudo-labeling:** 0.120-0.125 (0.005-0.010 improvement)
- **With all techniques:** 0.115-0.120 (0.010-0.015 improvement)

**Note:** 0.12973 is already in top ~10% of competition. Further improvements get harder.

For this competition:
- **0.13** = very strong
- **0.125** = top ~5–10%
- **<0.12** = leaderboard grinders / leakage tricks

---

## What NOT to Do Anymore

- ❌ More linear models
- ❌ More polynomial features
- ❌ More Optuna trials blindly
- ❌ Trust CV RMSE alone
- ❌ Random new models "just to try"

**You're past that phase.**

---

*For detailed strategy, see TODO_BREAKTHROUGH.md (archived in docs/)*

