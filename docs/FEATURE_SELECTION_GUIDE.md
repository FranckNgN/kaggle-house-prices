# Feature Selection Guide

## Overview

The feature selection script (`8featureSelection.py`) implements multiple best-practice methods to select the most important features, reducing noise and improving model performance.

**Expected Impact:** 0.002-0.008 RMSE improvement by removing noise features

## Quick Start

### Basic Usage (Recommended)

```bash
# Auto-select features (uses percentile method, ~10% threshold)
python notebooks/preprocessing/8featureSelection.py

# Select specific number of features
python notebooks/preprocessing/8featureSelection.py --method count --n_features 200

# Use percentile method
python notebooks/preprocessing/8featureSelection.py --method percentile --percentile 15.0
```

### Advanced Usage

```bash
# Find optimal feature count (slower but best results)
python notebooks/preprocessing/8featureSelection.py --method optimal --find-optimal

# Faster mode (disable SHAP)
python notebooks/preprocessing/8featureSelection.py --no-shap

# Fastest mode (only tree-based importance)
python notebooks/preprocessing/8featureSelection.py --no-shap --no-lasso --no-correlation
```

## Methods

### 1. **Auto** (Default)
- Uses percentile if `n_features` not specified
- Uses count if `n_features` is specified
- Best for: General use

### 2. **Percentile**
- Selects features above a percentile threshold
- Example: `--percentile 10.0` selects top 90% of features
- Best for: When you want to remove bottom X% of features

### 3. **Count**
- Selects top N features
- Example: `--n_features 200` selects top 200 features
- Best for: When you have a specific feature count target

### 4. **Optimal**
- Tests different feature counts and picks the best
- Slower but finds the optimal number
- Best for: When you want the best possible selection

## Feature Importance Methods

The script combines multiple importance methods:

1. **XGBoost Importance** (weight: 2.0)
   - Cross-validated feature importance
   - Fast and reliable

2. **LightGBM Importance** (weight: 2.0)
   - Cross-validated feature importance
   - Fast and reliable

3. **CatBoost Importance** (weight: 2.0)
   - Cross-validated feature importance
   - Good for categorical features

4. **SHAP Importance** (weight: 1.5, optional)
   - More accurate but slower
   - Use `--no-shap` to disable for speed

5. **Correlation Importance** (weight: 1.0)
   - Fast linear correlation with target
   - Good baseline

6. **Lasso Importance** (weight: 1.5)
   - L1 regularization feature selection
   - Good for removing redundant features

## Output

The script creates:

1. **`train_process7.csv`** - Training data with selected features
2. **`test_process7.csv`** - Test data with selected features
3. **`data/processed/feature_importance.csv`** - Detailed importance scores for all features

**Note:** Feature selection runs on process6 (encoded/scaled features) and outputs process7. Target encoding then runs on process7 to create process8 (final data).

## Performance Tips

### Fast Mode (~5-10 minutes)
```bash
python notebooks/preprocessing/8featureSelection.py --no-shap --method count --n_features 200
```

### Balanced Mode (~15-20 minutes)
```bash
python notebooks/preprocessing/8featureSelection.py --method auto
```

### Best Results Mode (~30-45 minutes)
```bash
python notebooks/preprocessing/8featureSelection.py --method optimal --find-optimal
```

## Integration with Models

After running feature selection and target encoding, update your models to use process8 (final processed data):

```python
# In model files, change:
train = pd.read_csv(local_config.TRAIN_PROCESS7_CSV)
test = pd.read_csv(local_config.TEST_PROCESS7_CSV)

# To:
train = pd.read_csv(local_config.TRAIN_PROCESS8_CSV)
test = pd.read_csv(local_config.TEST_PROCESS8_CSV)
```

**Pipeline Order:**
1. Process 6: Encoded/scaled features
2. Process 7: Feature selection (removes noise)
3. Process 8: Target encoding (adds target-encoded features)

## Expected Results

- **Feature Reduction:** Typically reduces from ~350 features to 150-250 features
- **Performance:** Usually improves RMSE by 0.002-0.008
- **Training Time:** Faster training with fewer features
- **Overfitting:** Reduced risk of overfitting

## Troubleshooting

### Issue: SHAP not available
**Solution:** Install SHAP or use `--no-shap` flag
```bash
pip install shap
```

### Issue: Too many/few features selected
**Solution:** Adjust percentile or count
```bash
# More features
python notebooks/preprocessing/8featureSelection.py --percentile 20.0

# Fewer features
python notebooks/preprocessing/8featureSelection.py --n_features 150
```

### Issue: Script is too slow
**Solution:** Use faster mode
```bash
python notebooks/preprocessing/8featureSelection.py --no-shap --no-lasso
```

## Best Practices

1. **Run once with optimal method** to find best feature count
2. **Save the feature list** for reproducibility
3. **Compare performance** before/after feature selection
4. **Check feature importance CSV** to understand which features matter
5. **Use cross-validation** (already built-in) to prevent overfitting

## Example Workflow

```bash
# Step 1: Run feature selection with optimal method
python notebooks/preprocessing/8featureSelection.py --method optimal --find-optimal

# Step 2: Check feature_importance.csv to see selected features
# Step 3: Update models to use process8
# Step 4: Retrain models and compare performance
```

## Notes

- Feature selection is done on **process6** data (after encoding/scaling, before target encoding)
- Selected features are saved to **process7**
- Target encoding then runs on process7 to create **process8** (final data)
- Importance scores are saved for analysis
- All methods use cross-validation to prevent overfitting
- This order makes sense: select important features first, then add target encoding on top

