# House Prices - Advanced Regression Techniques

## Project Overview

Kaggle competition project predicting house sale prices using a structured preprocessing pipeline and ensemble modeling approach. The project implements a 6-stage preprocessing pipeline followed by multiple machine learning models, with final predictions generated through model blending.

**Key Decision**: Target variable transformed to `logSP = log1p(SalePrice)` for better distribution properties and improved model performance.

---

## Preprocessing Pipeline

### Process 1: Cleaning (`1cleaning.ipynb`)

**Objective**: Handle missing values and normalize data

- **Missing Value Strategy**:
  - Numeric columns: Fill with `0`
  - Categorical columns: Replace `"NA"` and empty strings with `pd.NA`, then fill with `"<None>"`
- **Output**: `train_process1.csv`, `test_process1.csv`
- **Feature Summary**: Generated `feature_summary.csv` categorizing features as numeric (discrete/continuous) or categorical

### Process 2: Data Engineering (`2dataEngineering.ipynb`)

**Objective**: Transform target variable and remove outliers

- **SalePrice Transformations**: Evaluated multiple transformations (log1p, sqrt, Box-Cox, Yeo-Johnson, Quantile)
- **Selected Transformation**: `logSP = log1p(SalePrice)` - provides best distribution properties
- **Outlier Removal**: Removed rows where `GrLivArea > 4000` AND `SalePrice < 300000` (2 outliers removed)
- **Output**: `train_process2.csv`, `test_process2.csv`
- **Note**: Original `SalePrice` column dropped, replaced with `logSP`

### Process 3: Skew/Kurtosis Normalization (`3skewKurtosis.ipynb`)

**Objective**: Normalize skewed numeric features

- **Method**: PowerTransformer with Yeo-Johnson method
- **Threshold**: Applied to numeric columns with `|skew| > 0.75`
- **Features Transformed**: `LotArea`, `MasVnrArea`, `BsmtFinSF1`, `BsmtFinSF2`, `BsmtUnfSF`, `1stFlrSF`, `2ndFlrSF`, `GrLivArea`, `GarageYrBlt`, `WoodDeckSF`, `OpenPorchSF`, `EnclosedPorch`, `ScreenPorch`
- **Output**: `train_process3.csv`, `test_process3.csv`

### Process 4: Feature Engineering (`4featureEngineering.ipynb`)

**Objective**: Create new features and apply clustering

- **Age Features**:
  - `Age = YrSold - YearBuilt`
  - `Garage_Age = YrSold - GarageYrBlt`
  - `RemodAge = YrSold - YearRemodAdd`
- **K-Means Clustering**:
  - Features: `GrLivArea`, `TotalBsmtSF`, `1stFlrSF`, `GarageCars`, `YearBuilt`, `OverallQual`
  - Clusters: `k=4`, standardized features before clustering
  - Output: `KMeansCluster` feature
- **Output**: `train_process4.csv`, `test_process4.csv`

### Process 5: Scaling (`5scaling.ipynb`)

**Objective**: Standardize continuous numeric features

- **Method**: StandardScaler
- **Scope**: Applied only to continuous numeric features (ratio of unique values > 5%)
- **Excluded**: Discrete numeric and categorical features remain unchanged
- **Output**: `train_process5.csv`, `test_process5.csv`

### Process 6: Categorical Encoding (`6categorialEncode.ipynb`)

**Objective**: Convert categorical features to numeric format

- **Method**: One-hot encoding with `drop_first=True`
- **Implementation**: Concatenate train and test sets before encoding to ensure consistent dummy variables
- **Data Type**: Encoded features stored as `int8` for memory efficiency
- **Output**: `train_process6.csv`, `test_process6.csv` (final processed datasets)

---

## Models Implemented

All models use `logSP` as target and transform predictions back to original scale using `expm1()`.

### Linear Models

#### 1. Linear Regression (`1linearRegUpdated.ipynb`)
- **Method**: Standard OLS regression
- **Cross-Validation**: 5-fold KFold
- **Evaluation**: RMSE on real prices (after exp transformation)
- **CV RMSE**: ~23,240 (mean), std: ~2,491
- **Kaggle Performance**: RMSLE 0.15051 (baseline: 0.16538, Rank 4315)
- **Output**: `linearModelUpdated.csv`

#### 2. Ridge Regression (`2ridgeModel.ipynb`)
- **Regularization**: L2 penalty
- **Hyperparameter Search**: GridSearchCV over alpha values `[0.01, 0.1, 1-30]`
- **Cross-Validation**: 5-fold KFold
- **Best Alpha**: 13
- **CV RMSE**: 20,562.79
- **Evaluation**: Custom scorer - RMSE on real prices
- **Kaggle Performance**: RMSLE 0.13272, Rank 2167
- **Output**: `ridgeModel.csv`

#### 3. Lasso Regression (`3lassoModel.ipynb`)
- **Regularization**: L1 penalty
- **Hyperparameter Search**: GridSearchCV over alpha values `[0.01, 0.1, 1-30]`
- **Cross-Validation**: 5-fold KFold
- **Best Alpha**: 0.01
- **CV RMSE**: 24,120.01
- **Max Iterations**: 10,000
- **Kaggle Performance**: RMSLE 0.14850
- **Output**: `lassoModel.csv`

#### 4. Elastic Net (`4elasticNetModel.ipynb`)
- **Regularization**: Combined L1 + L2 penalty
- **Hyperparameter Search**: Manual grid search
  - Alpha: `[0.0001, 0.0005, 0.0007, 0.0008, 0.0009, 0.001, 0.0011, 0.0012, 0.0013, 0.0015, 0.002, 0.005, 0.01]`
  - L1 Ratio: `[0.1, 0.3, 0.5, 0.7, 0.9]`
- **Best Parameters**: Found through exhaustive search (α=0.0008, l1_ratio=0.9)
- **Max Iterations**: 20,000
- **Kaggle Performance**: RMSLE 0.13677, Rank 2567
- **Output**: `elasticNetModel.csv`

### Tree-Based Models

#### 5. XGBoost (`5XGBoostModel.ipynb`)
- **Method**: XGBRegressor with GPU acceleration
- **Hyperparameter Search**: RandomizedSearchCV (30 iterations)
- **Search Space**:
  - `n_estimators`: [800, 1000, 1200]
  - `learning_rate`: [0.05, 0.04, 0.03]
  - `max_depth`: [3, 4, 5]
  - `subsample`: [0.8, 1.0]
  - `colsample_bytree`: [0.7, 0.9]
- **Best Parameters**: `subsample=0.8`, `n_estimators=800`, `max_depth=3`, `learning_rate=0.03`, `colsample_bytree=0.7`
- **CV RMSE**: 0.1155 (log space)
- **Device**: CUDA GPU
- **Kaggle Performance**: RMSLE 0.13125, Rank 1215
- **Output**: `xgboost_Model.csv`

#### 6. LightGBM (`6lightGbmModel.ipynb`)
- **Method**: LGBMRegressor with GPU acceleration
- **Hyperparameter Search**: RandomizedSearchCV (30 iterations, 5-fold CV)
- **Search Space**:
  - `num_leaves`: [31, 63]
  - `max_depth`: [5, 7, -1]
  - `learning_rate`: [0.05, 0.04, 0.03]
  - `n_estimators`: [800, 1000, 1200]
  - `subsample`: [0.8, 1.0]
  - `colsample_bytree`: [0.7, 0.9]
  - `min_child_samples`: [10, 20]
- **Device**: GPU
- **Kaggle Performance**: RMSLE 0.12647, Rank 1215
- **Output**: `lightGBM_Model.csv`

#### 7. CatBoost (`7catBoostModel.ipynb`)
- **Method**: CatBoostRegressor with GPU acceleration
- **Hyperparameter Search**: Manual grid search with 5-fold CV
- **Search Space**:
  - `learning_rate`: [0.03]
  - `depth`: [6]
  - `l2_leaf_reg`: [3.0, 5.0]
  - `iterations`: [2000]
- **Best Parameters**: `learning_rate=0.03`, `depth=6`, `l2_leaf_reg=3.0`, `iterations=2000`
- **Best CV RMSE**: 0.12054 (log space)
- **Device**: GPU (device 0)
- **Output**: `catboost_Model.csv`

### Ensemble

#### 8. Blending Model (`8blendingModel.py`)
- **Method**: Weighted average of top 3 tree-based models
- **Weights**:
  - XGBoost: 2.0
  - LightGBM: 0.5
  - CatBoost: 1.0
- **Formula**: `(2.0 * XGB + 0.5 * LGB + 1.0 * CAT) / 3.5`
- **Output**: `blend_xgb_lgb_cat_Model.csv`

---

## Key Technical Decisions

### Target Variable Transformation
- **Choice**: `log1p(SalePrice)` instead of raw `SalePrice`
- **Rationale**: 
  - SalePrice follows log-normal distribution
  - Reduces impact of outliers
  - Improves model performance on skewed data
  - Standard practice in regression competitions

### Outlier Handling
- **Strategy**: Domain-specific outlier removal
- **Criteria**: `GrLivArea > 4000` AND `SalePrice < 300000`
- **Result**: 2 outliers removed from training set

### Cross-Validation Strategy
- **Method**: 5-fold KFold with `shuffle=True`, `random_state=42`
- **Consistency**: All models use same CV strategy for fair comparison
- **Evaluation**: RMSE on real prices (after inverse log transformation)

### Feature Engineering Approach
- **Age Features**: Capture temporal relationships (house age, garage age, remodel age)
- **Clustering**: K-means on key features to capture non-linear patterns
- **Skew Normalization**: Yeo-Johnson transformation for skewed numeric features

### Scaling Strategy
- **Selective Scaling**: Only continuous numeric features scaled
- **Preservation**: Discrete numeric and categorical features remain in original scale
- **Method**: StandardScaler (zero mean, unit variance)

### Categorical Encoding
- **Method**: One-hot encoding with `drop_first=True`
- **Alignment**: Train and test sets encoded together to ensure consistent dummy variables
- **Memory**: Encoded as `int8` for efficiency

---

## Project Structure

```
house-prices-starter/
├── data/
│   ├── raw/                    # Original Kaggle data
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── data_description.txt
│   ├── interim/                # Intermediate processed files
│   │   ├── train_process1.csv  # After cleaning
│   │   ├── train_process2.csv  # After data engineering
│   │   ├── train_process3.csv  # After skew normalization
│   │   ├── train_process4.csv  # After feature engineering
│   │   ├── train_process5.csv  # After scaling
│   │   └── train_process6.csv  # After encoding (final)
│   ├── processed/              # Final processed features
│   │   ├── feature_summary.csv
│   │   └── salePrice_transforms.csv
│   └── submissions/           # Kaggle submission files
│       ├── xgboost_Model.csv
│       ├── lightGBM_Model.csv
│       ├── catboost_Model.csv
│       └── blend_xgb_lgb_cat_Model.csv
├── notebooks/
│   ├── preprocessing/         # Preprocessing pipeline
│   │   ├── 1cleaning.ipynb
│   │   ├── 2dataEngineering.ipynb
│   │   ├── 3skewKurtosis.ipynb
│   │   ├── 4featureEngineering.ipynb
│   │   ├── 5scaling.ipynb
│   │   ├── 6categorialEncode.ipynb
│   │   └── run_preprocessing.py  # Automated pipeline execution
│   └── Models/                 # Model training notebooks
│       ├── 0linearRegression.ipynb
│       ├── 1linearRegUpdated.ipynb
│       ├── 2ridgeModel.ipynb
│       ├── 3lassoModel.ipynb
│       ├── 4elasticNetModel.ipynb
│       ├── 5XGBoostModel.ipynb
│       ├── 6lightGbmModel.ipynb
│       ├── 7catBoostModel.ipynb
│       └── 8blendingModel.py
├── config_local/              # Configuration management
│   └── local_config.py        # Centralized path configuration
├── utils/                     # Utility functions
│   └── checks.py              # Data validation checks
└── tests/                     # Integration tests
    └── test_checks_integration.py
```

---

## Automation

### Preprocessing Pipeline
- **Script**: `notebooks/preprocessing/run_preprocessing.py`
- **Method**: Uses Papermill to execute notebooks sequentially
- **Functionality**: 
  - Cleans interim directory before execution
  - Auto-detects numbered notebooks (1-6)
  - Executes in order, overwriting notebooks in-place

### Configuration Management
- **File**: `config_local/local_config.py`
- **Purpose**: Centralized path management for all data files
- **Benefits**: Easy path updates, consistent across notebooks

---

## Current Status

✅ **Completed**:
- Full 6-stage preprocessing pipeline
- 7 individual models (4 linear, 3 tree-based)
- Ensemble blending model
- Automated preprocessing execution
- Configuration management system

📊 **Model Performance Summary**:
- **Best Linear Model**: Ridge (CV RMSE: 20,562.79 | Kaggle: RMSLE 0.13272, Rank 2167)
- **Best Tree Model**: LightGBM (Kaggle: RMSLE 0.12647, Rank 1215)
- **XGBoost**: Kaggle RMSLE 0.13125, Rank 1215
- **CatBoost**: CV RMSE 0.12054 (log space)
- **Ensemble**: Weighted blend of XGBoost, LightGBM, CatBoost

---

## Notes

- All models predict in log space (`logSP`) and transform back using `expm1()` for final submissions
- GPU acceleration used for tree-based models (XGBoost, LightGBM, CatBoost)
- Cross-validation consistently uses 5-fold KFold across all models
- Preprocessing pipeline is modular and can be re-run independently
- Feature engineering focused on temporal relationships and clustering patterns

