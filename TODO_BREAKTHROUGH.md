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

### 2️⃣ **Fix Ensemble Space Consistency** ⚠️ CRITICAL
**Problem**: Predictions explode to 1e17/1e60 because mixing log and real spaces.

**Root Cause**: 
- Some base models output `log(SalePrice)`
- Others output `SalePrice`
- Meta-model applies `expm1()` blindly

**Mandatory Fix** (non-negotiable):
- [ ] **Option A (Recommended)**: Log-space stacking
  - All base models trained on `log1p(SalePrice)`
  - Meta-model predicts in log space
  - Apply `expm1()` once at the very end
- [ ] **Option B**: Real-space stacking
  - Convert ALL base predictions with `expm1()` before stacking
  - Meta-model trained on real prices
  - No transform afterward
- [ ] Verify no numerical explosions
- [ ] Test on Kaggle and compare scores

**⚠️ Until this is clean, do not trust any ensemble result.**

---

### 3️⃣ **Fix CV Strategy** ⚠️ HIGH PRIORITY
**Problem**: CV is lying. Ridge CV RMSE ≈ 0.096 but Kaggle RMSLE ≈ 1.41.

**Why**: KFold splits mix cheap/expensive houses, but Kaggle test has different neighborhood composition.

**Action Items:**
- [ ] Implement GroupKFold or Stratified CV on target quantiles
- [ ] Bin SalePrice into deciles
- [ ] Stratify CV on those bins
- [ ] Retrain models with new CV strategy
- [ ] Compare: CV score should raise slightly, Kaggle score should improve

**Expected Impact**: CV must reflect Kaggle distribution, not academic purity.

---

### 4️⃣ **Increase Model Diversity** ⚠️ MEDIUM PRIORITY
**Problem**: Ensembles lack diversity (correlation > 0.95). XGB ↔ LGB ↔ CatBoost ≈ 0.96–0.98.

**Action Items:**
- [ ] **A. Different feature views**:
  - Train models on `process6` (raw one-hot)
  - Train models on `process8` (target-encoded)
  - Train models on `cb_raw` (new raw CatBoost set)
- [ ] **B. Different loss behavior**:
  - Try CatBoost with `loss_function="MAE"`
  - Try CatBoost with `loss_function="Quantile:alpha=0.9"`
- [ ] **C. Remove Ridge from ensembles**:
  - Ridge dominating blending is a red flag
  - It correlates poorly with Kaggle → remove it entirely
- [ ] Measure correlation between new models
- [ ] Retrain ensembles with diverse base models

**Expected Impact**: Real diversity, not averaging the same opinion.

---

### 5️⃣ **Error-Driven Feature Engineering** ⚠️ HIGH PRIORITY
**Problem**: You've done feature engineering by intuition. Now do it by failure analysis.

**Action Items:**
- [ ] Take best CatBoost model and inspect:
  - Worst 5% predictions
  - Group errors by Neighborhood
  - Group errors by OverallQual
  - Group errors by YearBuilt buckets
- [ ] Identify patterns:
  - New houses overpriced?
  - Certain neighborhoods systematically underpredicted?
  - Remodel age interacting with quality?
- [ ] Add 3–5 targeted features:
  - `Is_NewHouse = YearBuilt > 2000`
  - `Neighborhood_Median_Adjustment`
  - `Qual_Age_Interaction`
- [ ] Retrain and validate

**Expected Impact**: 3–5 targeted features > 50 generic ones. This is how you break 0.125.

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

1. **Day 1-2**: Create `process_cb_raw.csv` and retrain CatBoost
2. **Day 2-3**: Fix ensemble space consistency (log vs real)
3. **Day 3-4**: Implement better CV strategy (GroupKFold/Stratified)
4. **Day 4-5**: Error analysis and targeted feature engineering
5. **Week 2**: Model diversity improvements and pseudo-labeling

---

## 📝 Notes

- Current best: CatBoost 0.12973 (Kaggle)
- Current CV best: Ridge 0.09614 (but overfits severely)
- Ensemble status: Fixed numerical explosions but underperform single models
- Key insight: Simpler is better for CatBoost. Stop over-engineering inputs.

---

**Last Updated**: 2025-12-20

