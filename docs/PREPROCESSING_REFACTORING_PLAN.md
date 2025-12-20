# Preprocessing Pipeline Refactoring Plan

## Critical Issues Found

### 1. Data Leakage (CRITICAL)
- **Issue**: `add_neighborhood_price_stats` in `4featureEngineering.py` uses target encoding BEFORE categorical encoding
- **Fix**: Remove from stage 4, integrate into `8targetEncoding.py` where it belongs
- **Impact**: Prevents data leakage, improves leaderboard score

### 2. Order Issues
- **Issue**: Polynomial features created before scaling - should be after or scaled separately
- **Fix**: Create polynomial features after scaling, or scale them separately
- **Impact**: Better feature scaling, improved model performance

### 3. Missing Validation
- **Issue**: No inline validation checks in individual files
- **Fix**: Add validation checks using `utils.checks` module
- **Impact**: Early error detection, better debugging

### 4. Feature Engineering Order
- **Issue**: Some features created before they're needed (e.g., ratios before scaling)
- **Fix**: Reorder feature creation to logical sequence
- **Impact**: Cleaner pipeline, easier to maintain

## Refactoring Plan

### Stage 1: Cleaning (1cleaning.py)
**Status**: ✅ Mostly good
**Changes**:
- Add basic validation (check for required columns)
- Improve error messages
- Add data type validation

### Stage 2: Data Engineering (2dataEngineering.py)
**Status**: ⚠️ Needs improvement
**Changes**:
- Add validation for outlier removal (log removed rows)
- Validate basic features are created correctly
- Ensure target transformation is correct
- Add shape consistency checks

### Stage 3: Skew Normalization (3skewKurtosis.py)
**Status**: ✅ Good
**Changes**:
- Add validation for transformed features
- Log which columns were transformed vs skipped
- Validate no infinite values after transformation

### Stage 4: Feature Engineering (4featureEngineering.py)
**Status**: 🔴 CRITICAL - Needs major refactoring
**Changes**:
- **REMOVE** `add_neighborhood_price_stats` (move to stage 8)
- Reorder feature creation:
  1. Basic features (era, flags, ordinal encoding)
  2. Group benchmarks (using train data only - already correct)
  3. Basic interactions
  4. Ratio features (before scaling - these will be scaled)
  5. Temporal features
  6. Quality aggregates
  7. Basic K-means clustering
  8. Advanced clustering (after more features exist)
  9. Advanced interactions (after aggregates exist)
- Add validation for each feature group
- Ensure no target leakage in any feature

### Stage 5: Scaling (5scaling.py)
**Status**: ⚠️ Needs improvement
**Changes**:
- Add validation that only continuous features are scaled
- Log which features were scaled
- Ensure binary features are NOT scaled (already correct)
- Validate scaling was applied correctly

### Stage 6: Categorical Encoding (6categorialEncode.py)
**Status**: ✅ Good
**Changes**:
- Add validation for encoding (check all categories handled)
- Validate train/test concatenation was done correctly
- Ensure no target leakage

### Stage 7: Feature Selection (7featureSelection.py)
**Status**: ✅ Good (newly created)
**Changes**:
- Add validation that selected features exist in both train/test
- Validate feature importance scores are reasonable
- Ensure no target leakage in selection process

### Stage 8: Target Encoding (8targetEncoding.py)
**Status**: ⚠️ Needs improvement
**Changes**:
- **ADD** `add_neighborhood_price_stats` from stage 4
- Add validation for target encoding (check CV was done correctly)
- Ensure no leakage in encoding process
- Validate encoded features are reasonable

## Implementation Order

1. **CRITICAL**: Fix data leakage in stage 4 (remove neighborhood stats)
2. **HIGH**: Add neighborhood stats to stage 8
3. **HIGH**: Add validation to all stages
4. **MEDIUM**: Reorder feature creation in stage 4
5. **MEDIUM**: Improve error handling throughout
6. **LOW**: Add logging improvements

## Expected Improvements

- **Data Leakage**: Eliminated → Better leaderboard score
- **Feature Quality**: Improved → Better model performance
- **Pipeline Robustness**: Improved → Fewer errors, easier debugging
- **Maintainability**: Improved → Clearer code structure

