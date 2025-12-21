# TODO: Breakthrough Strategy - Moving from 0.129 to <0.125 RMSLE

**Current Status**: Plateaued at ~0.129–0.130 RMSLE  
**Target**: <0.125 RMSLE (top ~5–10%)  
**Date**: 2025-12-20

---

## 🎯 Priority Actions (What Actually Moves the Needle)

### 1️⃣ **Simplify for CatBoost** ⚠️ HIGH PRIORITY
**Problem**: You engineered 248–264 features, then target-encoded, selected, scaled, etc.  
**Reality**: CatBoost does NOT need this. It's best with raw categoricals and simple numerics.

**Action Items:**
- [ ] Create `process_cb_raw.csv` pipeline:
  - ✅ Keep: Original categorical columns (Neighborhood, Exterior1st, etc.)
  - ✅ Keep: Age features (Age, Garage_Age, RemodAge)
  - ✅ Keep: Aggregate features (TotalSF, TotalBath)
  - ❌ Drop: One-hot encoding
  - ❌ Drop: Target encoding
  - ❌ Drop: Scaling
  - ❌ Drop: Polynomial features
  - ❌ Drop: Ratio features
  - ❌ Drop: KMeans clusters
- [ ] Retrain CatBoost on simplified pipeline
- [ ] Compare with current best (0.12973)

**Expected Impact**: Many top Kaggle solutions score better with fewer features in CatBoost.

---

### 2️⃣ **Fix Ensemble Space Consistency** ⚠️ CRITICAL ✅ COMPLETED
**Problem**: Predictions explode to 1e17/1e60 because mixing log and real spaces.

**Root Cause**: 
- Some base models output `log(SalePrice)`
- Others output `SalePrice`
- Meta-model applies `expm1()` blindly

**Mandatory Fix** (non-negotiable):
- [x] **Option A (Recommended)**: Log-space stacking ✅
  - All base models trained on `log1p(SalePrice)`
  - Meta-model predicts in log space
  - Apply `expm1()` once at the very end
- [x] Verify no numerical explosions ✅
- [x] Test blending model ✅ (predictions $51k-$545k, mean $178k - validated)

**Status**: ✅ **FIXED** - Blending model validated, no explosions, predictions in reasonable range

---

### 3️⃣ **Fix CV Strategy** ⚠️ HIGH PRIORITY ✅ COMPLETED
**Problem**: CV is lying. Ridge CV RMSE ≈ 0.096 but Kaggle RMSLE ≈ 1.41.

**Why**: KFold splits mix cheap/expensive houses, but Kaggle test has different neighborhood composition.

**Action Items:**
- [x] Implement GroupKFold or Stratified CV on target quantiles ✅
- [x] Bin SalePrice into deciles ✅
- [x] Stratify CV on those bins ✅
- [ ] Retrain models with new CV strategy (pending - ready to retrain)
- [ ] Compare: CV score should raise slightly, Kaggle score should improve (pending)

**Status**: ✅ **IMPLEMENTED** - Created `utils/cv_strategy.py`, updated stacking and optimization utilities. Ready for model retraining.

---

### 4️⃣ **Increase Model Diversity** ⚠️ MEDIUM PRIORITY 🔄 IN PROGRESS
**Problem**: Ensembles lack diversity (correlation > 0.95). XGB ↔ LGB ↔ CatBoost ≈ 0.96–0.98.

**Action Items:**
- [ ] **A. Different feature views**:
  - Train models on `process6` (raw one-hot)
  - Train models on `process8` (target-encoded) - ✅ Ready (just regenerated)
  - Train models on `cb_raw` (new raw CatBoost set)
- [ ] **B. Different loss behavior**:
  - Try CatBoost with `loss_function="MAE"`
  - Try CatBoost with `loss_function="Quantile:alpha=0.9"`
- [x] **C. Remove Ridge from ensembles** ✅:
  - Ridge dominating blending is a red flag
  - It correlates poorly with Kaggle → remove it entirely ✅ DONE
- [ ] Measure correlation between new models
- [ ] Retrain ensembles with diverse base models

**Status**: 🔄 **IN PROGRESS** - Ridge removed from ensembles. Ready to train on different feature sets and try different loss functions.

---

### 5️⃣ **Error-Driven Feature Engineering** ⚠️ HIGH PRIORITY ✅ COMPLETED
**Problem**: You've done feature engineering by intuition. Now do it by failure analysis.

**Action Items:**
- [x] Take best CatBoost model and inspect ✅:
  - Worst 5% predictions ✅ (47.63% error, mean $63k error)
  - Group errors by Neighborhood ✅
  - Group errors by OverallQual ✅
  - Group errors by YearBuilt buckets ✅
- [x] Identify patterns ✅:
  - Old houses (YearBuilt < 1960): 14.67% error ✅
  - New houses (YearBuilt > 2005): 9.69% error ✅
  - Low quality (OverallQual < 5): 9.88% error ✅
  - Remodel age interacting with quality ✅
- [x] Add 3–5 targeted features ✅:
  - `Qual_Age_Interaction` = `OverallQual * (YrSold - YearBuilt)` ✅
  - `RemodAge_FromBuild` = `YearRemodAdd - YearBuilt` ✅
  - `Is_Remodeled` = `(YearRemodAdd != YearBuilt)` ✅
  - `OverallQual_Squared` = `OverallQual ** 2` ✅
- [x] Features implemented in preprocessing ✅
- [x] Preprocessing pipeline re-run with new features ✅
- [ ] Retrain CatBoost and validate (pending)

**Status**: ✅ **COMPLETED** - All 4 features implemented and included in process8 data (254 features). Ready to retrain models.

---

### 6️⃣ **Pseudo-Labeling** ⚠️ ADVANCED (After Pipeline Stable)
**Action Items:**
- [ ] Predict on test with best CatBoost
- [ ] Select top confidence predictions (low variance across folds)
- [ ] Add to training with low weight
- [ ] Retrain
- [ ] Validate improvement

**Expected Impact**: Often gives 0.002–0.004 RMSLE improvement.

**⚠️ Only do this after pipeline is stable.**

---

## ❌ What NOT to Do Anymore

- ❌ More linear models
- ❌ More polynomial features
- ❌ More Optuna trials blindly
- ❌ Trust CV RMSE alone
- ❌ Random new models "just to try"

**You're past that phase.**

---

## 📊 Realistic Expectations

For this competition:
- **0.13** = very strong
- **0.125** = top ~5–10%
- **<0.12** = leaderboard grinders / leakage tricks

**You are on the edge of 0.125, not missing fundamentals.**

---

## 🎯 Immediate Next Steps (This Week)

1. **Day 1-2**: Create `process_cb_raw.csv` and retrain CatBoost (pending - user preference)
2. **Day 2-3**: Fix ensemble space consistency (log vs real) ✅ **DONE**
3. **Day 3-4**: Implement better CV strategy (GroupKFold/Stratified) ✅ **DONE**
4. **Day 4-5**: Error analysis and targeted feature engineering ✅ **DONE**
5. **Week 2**: Model diversity improvements and pseudo-labeling (in progress)

## ✅ Completed Today (2025-12-20)

1. ✅ Fixed ensemble space consistency (blending model)
2. ✅ Implemented stratified CV strategy
3. ✅ Removed Ridge from ensembles
4. ✅ Created and ran error analysis tool
5. ✅ Implemented 4 error-driven features
6. ✅ Re-ran full preprocessing pipeline with new features
7. ✅ Validated all preprocessing stages (all checks passed)
8. ✅ Tested and validated blending model

## 🔄 Next Actions (Ready to Execute)

1. **Retrain CatBoost** with new features (254 features, includes error-driven features)
2. **Retrain XGBoost/LightGBM** with stratified CV
3. **Test improved ensembles** on Kaggle
4. **Compare performance** with previous best (0.12973)

---

## 📝 Notes

- Current best: CatBoost 0.12973 (Kaggle)
- Current CV best: Ridge 0.09614 (but overfits severely)
- Ensemble status: ✅ Fixed numerical explosions, validated ($51k-$545k range, mean $178k)
- Key insight: Simpler is better for CatBoost. Stop over-engineering inputs.
- **New features added**: 4 error-driven features implemented (Qual_Age_Interaction, RemodAge_FromBuild, Is_Remodeled, OverallQual_Squared)
- **Preprocessing**: Full pipeline re-run with new features (254 features in process8)
- **Validation**: All sanity checks passed for all 8 stages

---

**Last Updated**: 2025-12-20 (Updated with completion status)

