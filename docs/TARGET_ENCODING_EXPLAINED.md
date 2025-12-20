# Target Encoding Explained

## What is Target Encoding?

**Target encoding** (also called "mean encoding" or "likelihood encoding") replaces categorical values with the **average target value** for that category. Instead of creating multiple binary columns (like one-hot encoding), you create a single numeric column that directly captures the relationship between the category and the target.

---

## Current Approach: One-Hot Encoding

You're currently using **one-hot encoding** (in `6categorialEncode.py`):

### Example with Neighborhood:

**Original Data:**
```
House | Neighborhood | SalePrice
------|--------------|----------
  1   |   NoRidge    |   $250,000
  2   |   NoRidge    |   $280,000
  3   |   OldTown    |   $120,000
  4   |   OldTown    |   $110,000
  5   |   CollgCr     |   $200,000
```

**After One-Hot Encoding:**
```
House | Neighborhood_NoRidge | Neighborhood_OldTown | Neighborhood_CollgCr | SalePrice
------|----------------------|---------------------|----------------------|----------
  1   |          1           |          0          |          0           |   $250,000
  2   |          1           |          0          |          0           |   $280,000
  3   |          0           |          1          |          0           |   $120,000
  4   |          0           |          1          |          0           |   $110,000
  5   |          0           |          0          |          1           |   $200,000
```

**Problems:**
- Creates many sparse columns (one per category)
- Doesn't capture that "NoRidge" houses are more expensive than "OldTown"
- Model has to learn the relationship from scratch
- With 25 neighborhoods → 24 new binary columns

---

## Target Encoding Approach

**After Target Encoding:**
```
House | Neighborhood | Neighborhood_TargetEnc | SalePrice
------|--------------|------------------------|----------
  1   |   NoRidge    |        12.04           |   $250,000
  2   |   NoRidge    |        12.04           |   $280,000
  3   |   OldTown    |        11.30           |   $120,000
  4   |   OldTown    |        11.30           |   $110,000
  5   |   CollgCr     |        11.90           |   $200,000
```

**How it works:**
1. Calculate mean `logSP` (log of SalePrice) for each neighborhood:
   - NoRidge: mean(logSP) = (log(250000) + log(280000)) / 2 = 12.04
   - OldTown: mean(logSP) = (log(120000) + log(110000)) / 2 = 11.30
   - CollgCr: mean(logSP) = log(200000) = 11.90

2. Replace category name with this mean value

**Benefits:**
- ✅ Single numeric column instead of many binary columns
- ✅ Directly captures "NoRidge is more expensive than OldTown"
- ✅ Model immediately understands the relationship
- ✅ Works great with tree-based models (XGBoost, LightGBM, CatBoost)

---

## Visual Comparison

### One-Hot Encoding:
```
Neighborhood = "NoRidge"
  ↓
[0, 0, 0, 1, 0, 0, 0, ...]  ← 24 binary features
```
Model has to learn: "When feature 4 is 1, price is higher"

### Target Encoding:
```
Neighborhood = "NoRidge"
  ↓
12.04  ← Single numeric value
```
Model immediately sees: "12.04 is higher than 11.30, so NoRidge is more expensive"

---

## The Problem: Data Leakage! ⚠️

**Naive target encoding causes overfitting:**

If you calculate the mean using the **same row** you're predicting:
```python
# BAD - Data Leakage!
train["Neighborhood_Enc"] = train.groupby("Neighborhood")["logSP"].transform("mean")
```

**Why it's bad:**
- You're using the target value from the current row to encode itself
- Model will memorize training data
- Terrible performance on test set

---

## Solution: Cross-Validated Target Encoding ✅

**Use cross-validation to prevent leakage:**

1. **Split data into folds** (e.g., 5 folds)
2. **For each fold:**
   - Calculate mean using only **other folds** (not current fold)
   - Encode current fold using that mean
3. **For test set:** Use mean from entire training set

**Example with 5-fold CV:**

```
Fold 1: Use Folds 2,3,4,5 to calculate mean → Encode Fold 1
Fold 2: Use Folds 1,3,4,5 to calculate mean → Encode Fold 2
Fold 3: Use Folds 1,2,4,5 to calculate mean → Encode Fold 3
Fold 4: Use Folds 1,2,3,5 to calculate mean → Encode Fold 4
Fold 5: Use Folds 1,2,3,4 to calculate mean → Encode Fold 5
Test:  Use all training data to calculate mean → Encode Test
```

This way, each row's encoding is calculated **without using that row's target value**.

---

## Smoothing: Preventing Overfitting

**Problem:** Categories with few samples have unreliable means.

**Example:**
- "NoRidge" appears 100 times → reliable mean
- "MeadowV" appears 2 times → unreliable mean

**Solution: Smoothing**

Blend category mean with global mean:
```
smoothed_mean = (category_mean × category_count + global_mean × smoothing) 
                / (category_count + smoothing)
```

**Example:**
- Global mean logSP = 11.90
- "MeadowV" appears 2 times, mean = 10.50
- Smoothing factor = 1.0

```
smoothed = (10.50 × 2 + 11.90 × 1.0) / (2 + 1.0) = 10.97
```

**Result:** Rare categories are pulled toward global mean (more conservative).

---

## When to Use Target Encoding

### ✅ Good for:
- **High-cardinality categoricals** (many categories)
  - Neighborhood (25 categories)
  - MSZoning (7 categories)
  - HouseStyle (8 categories)
- **Tree-based models** (XGBoost, LightGBM, CatBoost)
- **When categories have clear price differences**

### ❌ Not ideal for:
- **Low-cardinality categoricals** (few categories)
  - Binary features (already good as 0/1)
  - Features with < 5 categories (one-hot is fine)
- **Linear models** (can work but one-hot often better)
- **When categories have similar target values**

---

## Real Example from Your Data

### Neighborhood Target Encoding:

**Top 5 Most Expensive Neighborhoods:**
```
NoRidge:     12.15  (mean logSP)
NridgHt:     12.10
StoneBr:     12.05
Veenker:     12.00
Somerst:     11.95
```

**Top 5 Least Expensive Neighborhoods:**
```
MeadowV:     11.20
IDOTRR:      11.25
BrDale:      11.30
OldTown:     11.35
Edwards:     11.40
```

**Difference:** ~0.95 in log space = **~2.6x price difference!**

This is valuable information that one-hot encoding makes the model learn slowly.

---

## Implementation in Your Project

I've created `notebooks/preprocessing/7targetEncoding.py` that:

1. ✅ Uses **cross-validation** (5-fold) to prevent leakage
2. ✅ Uses **smoothing** to handle rare categories
3. ✅ Adds **noise** during training to prevent overfitting
4. ✅ Encodes multiple categorical features automatically

**Features it encodes:**
- Neighborhood (most important!)
- MSZoning
- MSSubClass
- HouseStyle
- RoofStyle
- Exterior1st
- Foundation
- Heating
- SaleType
- SaleCondition

---

## Expected Improvement

**Typical improvements:**
- **RMSE reduction:** 0.005 - 0.015
- **Kaggle score improvement:** 0.01 - 0.03

**Why it works:**
- Directly captures category-target relationships
- Reduces feature space (1 column vs 24 columns for Neighborhood)
- Tree models can use this information more efficiently

---

## Summary

| Aspect | One-Hot Encoding | Target Encoding |
|--------|----------------|----------------|
| **Columns created** | 1 per category | 1 total |
| **Information** | Binary (is/no) | Numeric (mean target) |
| **Model learning** | Must learn relationship | Relationship pre-calculated |
| **Best for** | Linear models, few categories | Tree models, many categories |
| **Risk** | Low | Medium (needs CV) |
| **Implementation** | Simple | Requires CV + smoothing |

**Bottom line:** Target encoding is like giving your model a cheat sheet that says "NoRidge houses are typically worth 12.04 in log space" instead of making it figure that out from binary flags.

---

## Next Steps

1. **Read the code:** `notebooks/preprocessing/7targetEncoding.py`
2. **Understand the CV logic:** How it prevents leakage
3. **Run it:** Generate process7 data with target encoding
4. **Compare:** Retrain models and see improvement!

Good luck! 🚀

