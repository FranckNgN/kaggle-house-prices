# Advanced Feature Engineering - Implementation Summary

## ✅ What Was Implemented

I've created `notebooks/preprocessing/8advancedFeatureEngineering.py` which adds **7 categories of advanced features** to improve your model performance.

---

## 📊 New Features Added

### 1. **Neighborhood Price Statistics** (5 features)
**Cross-validated target encoding** to prevent data leakage:
- `Neighborhood_mean_logSP` - Average log(SalePrice) by neighborhood
- `Neighborhood_median_logSP` - Median log(SalePrice) by neighborhood  
- `Neighborhood_std_logSP` - Standard deviation (price variability)
- `Neighborhood_min_logSP` - Minimum price in neighborhood
- `Neighborhood_max_logSP` - Maximum price in neighborhood

**Why it helps:** Directly captures location premium without one-hot encoding.

---

### 2. **Polynomial Features** (7 features)
**Squared terms** for key features:
- `TotalSF_squared`
- `OverallQual_squared`
- `GrLivArea_squared`
- `GarageArea_squared`
- `Age_squared`
- `LotArea_squared`
- `TotalBath_squared`

**Why it helps:** Captures non-linear relationships (e.g., larger houses have exponentially higher value).

---

### 3. **Ratio Features** (8 features)
**Efficiency and density metrics:**
- `LotArea_to_TotalSF_Ratio` - Lot efficiency
- `GarageArea_to_TotalSF_Ratio` - Garage proportion
- `Porch_to_TotalSF_Ratio` - Porch proportion
- `Living_to_Total_Ratio` - Living area efficiency
- `Bsmt_to_Total_Ratio` - Basement proportion
- `Rooms_per_SF` - Room density
- `Bedrooms_per_SF` - Bedroom density
- `Bath_per_SF` - Bathroom density

**Why it helps:** Captures efficiency metrics (e.g., a 2000 sqft house with 4 bedrooms is different from one with 2 bedrooms).

---

### 4. **Temporal Features** (5 features)
**Year and month effects:**
- `YearsSince2006` - Market trend over time
- `MarketCycle` - 4-year market cycle position
- `Quarter` - Seasonal quarter (1-4)
- `PeakSeason` - Spring/summer indicator (Mar-Aug)
- `EndOfYear` - Nov-Dec indicator

**Why it helps:** Captures market trends and seasonality effects.

---

### 5. **Quality Aggregate Features** (6 features)
**Aggregated quality metrics:**
- `AvgQuality` - Average quality across all quality features
- `MaxQuality` - Best quality feature
- `MinQuality` - Worst quality feature
- `QualityRange` - Quality consistency (max - min)
- `ExcellentFeatures` - Count of excellent features (score >= 4)
- `PoorFeatures` - Count of poor features (score <= 2)

**Why it helps:** Captures overall quality profile and consistency.

---

### 6. **Advanced Clustering** (6-8 features)
**Multiple K-Means clusters** with different k values:
- `Cluster_Size_k6`, `Cluster_Size_k8` - Size-based clusters
- `Cluster_Quality_k6`, `Cluster_Quality_k8` - Quality-based clusters
- `Cluster_Location_k4`, `Cluster_Location_k6` - Location-based clusters

**Why it helps:** Captures non-linear patterns and house type groupings.

---

### 7. **Advanced Interaction Features** (5 features)
**Sophisticated interactions:**
- `Qual_x_LotArea` - Quality × Lot size
- `Qual_x_GarageArea` - Quality × Garage size
- `Qual_x_Bath` - Quality × Bathroom count
- `Age_x_Condition` - Age × Condition (maintenance effect)
- `SF_x_Qual` - Size × Quality
- `Neighborhood_x_Qual` - Location × Quality (if neighborhood stats exist)

**Why it helps:** Captures multiplicative effects (e.g., high quality + large lot = premium).

---

## 📈 Expected Impact

**Total New Features:** ~40-45 features

**Expected Improvement:**
- **RMSE reduction:** 0.002 - 0.010
- **Kaggle score improvement:** 0.01 - 0.02

**Why it works:**
- More informative features capture relationships models struggle to learn
- Ratio features capture efficiency metrics
- Neighborhood stats provide location premium directly
- Quality aggregates summarize complex quality profiles

---

## 🚀 How to Use

### Step 1: Run Advanced Feature Engineering

```bash
python notebooks/preprocessing/8advancedFeatureEngineering.py
```

This will:
- Load `train_process6.csv` and `test_process6.csv`
- Add all advanced features
- Save to `train_process8.csv` and `test_process8.csv`

### Step 2: Update Model Scripts

Change your model scripts to use process8 data:

```python
# In notebooks/Models/7XGBoostModel.py, etc.
# Change:
train = pd.read_csv(local_config.TRAIN_PROCESS6_CSV)
test = pd.read_csv(local_config.TEST_PROCESS6_CSV)

# To:
train = pd.read_csv(local_config.TRAIN_PROCESS8_CSV)
test = pd.read_csv(local_config.TEST_PROCESS8_CSV)
```

### Step 3: Retrain Models

Retrain your best models:
- XGBoost
- LightGBM
- CatBoost
- Stacking

### Step 4: Compare Performance

Check `runs/model_performance.csv` to see improvements!

---

## ⚠️ Important Notes

1. **Neighborhood Stats:** Uses cross-validation to prevent data leakage. Safe to use!

2. **Feature Count:** You'll go from ~264 features to ~300+ features. Consider feature selection if needed.

3. **Clustering:** Some clustering features may not be created if required base features are missing. This is normal.

4. **Compatibility:** Works with your existing preprocessing pipeline (process6 → process8).

---

## 🔍 Feature Details

### Neighborhood Price Statistics
- **Method:** Cross-validated target encoding
- **Prevents leakage:** Yes (uses CV folds)
- **Smoothing:** Uses global statistics as fallback

### Polynomial Features
- **Method:** Simple squared terms
- **Features:** Key numeric features only
- **Risk:** Low (adds non-linearity)

### Ratio Features
- **Method:** Division with +1 to prevent division by zero
- **Handling:** Replaces inf/NaN with 0
- **Risk:** Low

### Temporal Features
- **Method:** Derived from YrSold and MoSold
- **Assumptions:** 4-year market cycles
- **Risk:** Low

### Quality Aggregates
- **Method:** Statistical aggregations across quality scores
- **Features:** All `_Score` columns + OverallQual/Cond
- **Risk:** Low

### Clustering
- **Method:** K-Means with StandardScaler
- **K values:** 4, 6, 8 (depending on config)
- **Risk:** Medium (may not help if base features are weak)

### Interaction Features
- **Method:** Multiplicative combinations
- **Features:** Quality × Size, Age × Condition, etc.
- **Risk:** Low

---

## 📊 Comparison: Before vs After

### Before (Process 6):
- **Features:** ~264
- **Encoding:** One-hot encoding
- **Interactions:** Basic (3 features)
- **Clustering:** Basic (k=4, one set)

### After (Process 8):
- **Features:** ~300-310
- **Encoding:** One-hot + target encoding (neighborhood stats)
- **Interactions:** Advanced (8+ features)
- **Clustering:** Advanced (multiple k values, different feature sets)
- **New:** Polynomial, ratio, temporal, quality aggregates

---

## 🎯 Next Steps

1. ✅ **Run the script** to generate process8 data
2. ✅ **Update model scripts** to use process8
3. ✅ **Retrain models** and compare performance
4. ✅ **Run optimized blending** with new features
5. ⚠️ **Consider feature selection** if too many features cause overfitting

---

## 💡 Tips

1. **Start with one model:** Test XGBoost first to see if features help
2. **Monitor overfitting:** Check CV score vs Kaggle score gap
3. **Feature importance:** Use SHAP to see which new features matter most
4. **Iterate:** Remove features that don't help

---

## 🐛 Troubleshooting

**Issue:** Script fails with "Neighborhood column not found"
- **Solution:** Make sure you're running on process6 data (after categorical encoding)

**Issue:** Too many features causing overfitting
- **Solution:** Use feature selection (SHAP importance or Lasso)

**Issue:** Clustering features not created
- **Solution:** Normal if base features are missing. Check which features exist.

**Issue:** Memory errors
- **Solution:** Process train and test separately, or use chunking

---

## 📚 References

- **Target Encoding:** See `docs/TARGET_ENCODING_EXPLAINED.md`
- **Improvement Roadmap:** See `docs/IMPROVEMENT_ROADMAP.md`
- **Quick Start:** See `docs/QUICK_START_IMPROVEMENTS.md`

---

Good luck! 🚀

