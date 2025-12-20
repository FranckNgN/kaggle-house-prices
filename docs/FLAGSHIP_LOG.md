# House Prices Prediction - Flagship Log

**A Comprehensive Machine Learning Pipeline for Real Estate Price Prediction**

*Project Journal & Showcase - Updated December 2025*

---

## Abstract

This project implements a complete machine learning pipeline for predicting house sale prices using advanced regression techniques. Through systematic 8-stage preprocessing, feature engineering, and ensemble modeling, we achieve **RMSLE 0.12973** (CatBoost, best score as of 2025-12-19) on the Kaggle leaderboard. The work demonstrates the critical importance of target transformation, feature engineering, model selection, and systematic validation in regression tasks.

---

## 1. Problem Statement & Vision

### Objective

Predict house sale prices $\hat{y}$ given features $\mathbf{X} \in \mathbb{R}^{n \times p}$:

$$\hat{y} = f(\mathbf{X}; \boldsymbol{\theta})$$

**Evaluation Metric**: Root Mean Squared Logarithmic Error (RMSLE)

$$\text{RMSLE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}[\log(1 + y_i) - \log(1 + \hat{y}_i)]^2}$$

### Vision

Create a robust, automated pipeline that:
- Handles real-world data complexities (missing values, skewness, categorical features)
- Leverages both linear and non-linear modeling approaches
- Optimizes hyperparameters systematically
- Provides reproducible and scalable solutions
- Includes comprehensive validation and sanity checks
- Enables easy model comparison and analysis

---

## 2. Core Hypothesis

### Target Distribution Hypothesis

**Hypothesis**: House sale prices follow a **log-normal distribution** rather than a normal distribution.

**Evidence**: 
- Right-skewed distribution in raw data
- Large variance in price ranges
- Standard practice in real estate and regression competitions

**Transformation**:
$$y_{log} = \log(1 + \text{SalePrice}) = \log(1 + y)$$

**Inverse for predictions**:
$$\hat{y} = \exp(\hat{y}_{log}) - 1$$

**Training Objective**: Minimize RMSE in log space
$$\mathcal{L} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_{log,i} - \hat{y}_{log,i})^2}$$

This transformation:
- Normalizes the target distribution
- Reduces impact of outliers
- Aligns with RMSLE evaluation metric
- Improves model convergence and performance

---

## 3. Methodology

### 3.1 Preprocessing Pipeline

An **8-stage systematic preprocessing approach** (enhanced from original 6 stages):

#### Stage 1: Cleaning
- **Missing Value Strategy**:
  - Numeric columns: Fill with `0`
  - Categorical columns: Replace `"NA"` and empty strings with `pd.NA`, then fill with `"<None>"`
- **Output**: `train_process1.csv`, `test_process1.csv`
- **Feature Summary**: Generated `feature_summary.csv` categorizing features

#### Stage 2: Data Engineering
- **Target Transformation**: `logSP = log1p(SalePrice)`
- **Outlier Removal**: Removed rows where `GrLivArea > 4000` AND `SalePrice < 300000` (2 outliers)
- **Output**: `train_process2.csv`, `test_process2.csv`

#### Stage 3: Skew/Kurtosis Normalization
- **Method**: PowerTransformer with Yeo-Johnson method
- **Threshold**: Applied to numeric columns with `|skew| > 0.75`
- **Features Transformed**: LotArea, MasVnrArea, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, 1stFlrSF, 2ndFlrSF, GrLivArea, GarageYrBlt, WoodDeckSF, OpenPorchSF, EnclosedPorch, ScreenPorch
- **Output**: `train_process3.csv`, `test_process3.csv`

#### Stage 4: Feature Engineering
- **Age Features**: `Age = YrSold - YearBuilt`, `Garage_Age = YrSold - GarageYrBlt`, `RemodAge = YrSold - YearRemodAdd`
- **Aggregate Features**: TotalSF, TotalBath, TotalPorchSF (log-normalized)
- **Group Benchmarks**: Neighborhood and MSSubClass ratios
- **Ordinal Scores**: Converted quality ratings to numeric (0-5)
- **Interaction Features**: Qual_x_TotalSF, Kitchen_x_TotalSF, Cond_x_Age
- **Polynomial Features**: Squared terms for key features (TotalSF, OverallQual, GrLivArea, etc.)
- **Ratio Features**: LotArea_to_TotalSF_Ratio, GarageArea_to_TotalSF_Ratio, etc.
- **K-means Clustering**: k=4 clusters on key features
- **Output**: `train_process4.csv`, `test_process4.csv`

#### Stage 5: Scaling
- **Method**: StandardScaler for continuous numeric features only
- **Preserves**: Categorical features unchanged
- **Output**: `train_process5.csv`, `test_process5.csv`

#### Stage 6: Categorical Encoding
- **Method**: Smart separation - one-hot encoding for low-cardinality (≤10 unique), kept as categorical for high-cardinality (>10 unique)
- **Output**: `train_process6.csv`, `test_process6.csv`
- **Result**: ~264 features after encoding

#### Stage 7: Feature Selection
- **Methods**: 
  - Cross-validated feature importance (XGBoost, LightGBM, CatBoost)
  - SHAP values (if available)
  - Permutation importance
  - Correlation-based selection
  - Lasso-based selection
- **Strategy**: Combines multiple methods for robustness
- **Output**: `train_process7.csv`, `test_process7.csv`
- **Result**: ~248-251 selected features (reduces from ~264)

#### Stage 8: Target Encoding
- **Method**: Cross-validated target encoding with smoothing
- **Features Encoded**: High-cardinality categorical features (Neighborhood, MSSubClass, etc.)
- **Prevents Overfitting**: Uses CV folds and noise injection during training
- **Output**: `train_process8.csv`, `test_process8.csv` (FINAL)
- **Result**: ~248-251 features ready for modeling

**Main Preprocessing Script**: `notebooks/preprocessing/run_preprocessing.py`
- Automatically runs all 8 stages sequentially
- Validates each stage before proceeding
- Tracks feature engineering summary

### 3.2 Model Selection & Formulations

#### Linear Models

**Ridge Regression** (L2 regularization):
$$\min_{\boldsymbol{\theta}} \|y_{log} - \mathbf{X}\boldsymbol{\theta}\|_2^2 + \alpha\|\boldsymbol{\theta}\|_2^2$$

**Lasso Regression** (L1 regularization):
$$\min_{\boldsymbol{\theta}} \|y_{log} - \mathbf{X}\boldsymbol{\theta}\|_2^2 + \alpha\|\boldsymbol{\theta}\|_1$$

**Elastic Net** (Combined L1 + L2):
$$\min_{\boldsymbol{\theta}} \|y_{log} - \mathbf{X}\boldsymbol{\theta}\|_2^2 + \alpha[\rho\|\boldsymbol{\theta}\|_1 + (1-\rho)\|\boldsymbol{\theta}\|_2^2]$$

#### Tree-Based Models

**Gradient Boosting** (XGBoost, LightGBM, CatBoost):
$$\mathcal{L} = \sum_{i=1}^{n} l(y_{log,i}, \hat{y}_{log,i}) + \sum_{k=1}^{K} \Omega(f_k)$$

where:
- $l$: loss function (RMSE)
- $\Omega(f_k) = \gamma T + \frac{1}{2}\lambda\|\mathbf{w}\|^2$: tree complexity penalty
- $K$: number of trees

**Random Forest**:
$$\hat{y} = \frac{1}{B}\sum_{b=1}^{B} T_b(\mathbf{X})$$

where $T_b$ are individual trees and $B$ is the number of trees.

#### Support Vector Regression

**Kernel-based Regression**:
$$\min_{\mathbf{w},b,\boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{n}(\xi_i + \xi_i^*)$$

subject to $|y_i - (\mathbf{w}^T\phi(\mathbf{x}_i) + b)| \leq \epsilon + \xi_i$.

**Kernel**: RBF (Radial Basis Function)

#### Ensemble Methods

**Blending** (Weighted Average):
$$\hat{y}_{blend} = \sum_{i} w_i \cdot \hat{y}_i$$

where weights are optimized using scipy.optimize (SLSQP) on OOF predictions.

**Stacking** (Meta-learner):
$$\hat{y}_{stack} = g(\hat{y}_{XGB}, \hat{y}_{LGB}, \hat{y}_{CAT}, \hat{y}_{Ridge}, \ldots)$$

where $g$ is a meta-model (Lasso with α=0.0005) trained on out-of-fold predictions from base models.

**Model Files** (in `notebooks/Models/`):
- `0linearRegression.py` - Linear Regression
- `1linearRegUpdated.py` - Linear Regression (Updated)
- `2ridgeModel.py` - Ridge Regression
- `3lassoModel.py` - Lasso Regression
- `4elasticNetModel.py` - Elastic Net
- `5randomForestModel.py` - Random Forest
- `6svrModel.py` - Support Vector Regression
- `7XGBoostModel.py` - XGBoost
- `8lightGbmModel.py` - LightGBM
- `9catBoostModel.py` - CatBoost
- `10blendingModel.py` - Blending Ensemble
- `11stackingModel.py` - Stacking Ensemble

**Main Model Training Script**: `scripts/run_all_models_parallel.py`
- Runs all models sequentially (0-11)
- Each model includes Optuna hyperparameter optimization
- Automatic validation and sanity checks
- Logs results to `runs/model_performance.csv`

### 3.3 Optimization Strategy

**Hyperparameter Search**: Bayesian optimization using Optuna (TPE sampler)
$$\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta} \in \Theta} \mathbb{E}[L_{CV}(\boldsymbol{\theta})]$$

**Cross-Validation**: 5-fold KFold for robust performance estimation
$$\text{CV-RMSE} = \frac{1}{5}\sum_{k=1}^{5} \text{RMSE}_k$$

**Runtime Optimization**: Target ~20 minutes per model with balanced search depth

### 3.4 Validation & Sanity Checks

**Comprehensive Validation System**:

1. **Preprocessing Validation** (`utils/checks.py`):
   - Raw data integrity checks
   - Train/test leakage detection
   - Fit/transform independence
   - Target leakage detection
   - Missing value handling verification
   - Shape consistency checks
   - Feature engineering sanity checks

2. **Model Validation** (`utils/model_wrapper.py`, `utils/model_validation.py`):
   - Input validation (X_train, y_train, X_test)
   - Prediction validation (NaN/Inf checks, range checks)
   - Submission format validation (columns, IDs, size)
   - Cross-validation split validation

3. **Automatic Wrappers**:
   - `validate_inputs_wrapper()` - Validates model inputs
   - `validate_predictions_wrapper()` - Validates predictions
   - `validate_submission_wrapper()` - Validates submission format
   - `validate_cv_wrapper()` - Validates CV splits

**Validation Control**: Can be enabled/disabled via `ENABLE_MODEL_VALIDATION` environment variable (default: enabled)

---

## 4. Results & Performance

This section presents a comprehensive analysis of all model experiments, following the iterative development process from baseline models through advanced ensembles. Results are organized chronologically to reflect the thought process and learning progression.

### 4.1 Experimental Evolution & Thought Process

The model development followed a systematic approach:

1. **Baseline Establishment** (Process6, 264 features): Established baseline performance with standard models
2. **Hyperparameter Optimization** (Process6): Applied Optuna Bayesian optimization to tree-based models
3. **Feature Engineering Enhancement** (Process8, 248-251 features): Integrated target encoding and refined feature selection
4. **Ensemble Exploration**: Attempted blending and stacking, revealing numerical stability challenges

### 4.2 Complete Model Performance Summary

#### 4.2.1 Tree-Based Models (Best Performers)

| Model | CV RMSE | Kaggle RMSLE | Features | Date | Status |
|-------|---------|--------------|----------|------|--------|
| **CatBoost** | **0.12017** | **0.12973** ⭐ | 264 (process6) | 2025-12-19 | ✅ Best Overall |
| CatBoost | 0.12064 | 0.13081 | 248 (process8) | 2025-12-20 | ✅ Latest |
| XGBoost | 0.11436 | - | 264 (process6) | 2025-12-19 | ✅ Best CV |
| XGBoost | 0.11864 | 0.13094 | 248 (process8) | 2025-12-20 | ✅ Latest |
| XGBoost | 0.11987 | 0.13335 | 264 (process6) | 2025-12-19 | ✅ Submitted |
| LightGBM | 0.11795 | - | 264 (process6) | 2025-12-19 | ✅ Good |
| LightGBM | 0.12097 | - | 248 (process8) | 2025-12-20 | ✅ Latest |
| Random Forest | 0.12749 | - | 248 (process8) | 2025-12-20 | ✅ Good |
| Random Forest | 0.13296 | 0.14460 | 264 (process6) | 2025-12-19 | ✅ Baseline |

**Key Observations:**
- **CatBoost** achieved best Kaggle score (0.12973) with process6 data, demonstrating superior categorical feature handling
- **XGBoost** achieved best CV RMSE (0.11436) but slightly higher Kaggle score, suggesting potential overfitting
- Process8 (with target encoding) shows comparable performance, indicating feature engineering trade-offs
- Tree-based models consistently outperform linear models by 0.01-0.02 RMSLE

#### 4.2.2 Linear Models

| Model | CV RMSE | Kaggle RMSLE | Features | Date | Status |
|-------|---------|--------------|----------|------|--------|
| Ridge | 0.09665 | - | 248 (process8) | 2025-12-20 | ✅ Best CV |
| Ridge | 0.11833 | 1.41358 | 264 (process6) | 2025-12-19 | ⚠️ Overfitting |
| Lasso | 0.20618 | - | 248 (process8) | 2025-12-20 | ⚠️ Moderate |
| Lasso | 0.30927 | 1.97336 | 264 (process6) | 2025-12-19 | ⚠️ Poor |
| Elastic Net | 0.19707 | - | 248 (process8) | 2025-12-20 | ⚠️ Moderate |
| Elastic Net | 0.30494 | 0.63422 | 264 (process6) | 2025-12-19 | ⚠️ Moderate |

**Key Observations:**
- Ridge shows excellent CV performance (0.09665) on process8 but severe overfitting (CV-Kaggle gap: 1.32)
- Linear models struggle with non-linear relationships in house price data
- L1/L2 regularization helps but cannot capture complex feature interactions

#### 4.2.3 Kernel Models

| Model | CV RMSE | Kaggle RMSLE | Features | Date | Status |
|-------|---------|--------------|----------|------|--------|
| SVR | 0.14561 | - | 248 (process8) | 2025-12-20 | ✅ Good |
| SVR | 0.15601 | 0.18191 | 264 (process6) | 2025-12-19 | ✅ Baseline |

**Key Observations:**
- SVR performs better than linear models but worse than tree-based models
- RBF kernel captures non-linearities but is computationally expensive

#### 4.2.4 Ensemble Models

| Model | CV RMSE | Kaggle RMSLE | Features | Date | Status |
|-------|---------|--------------|----------|------|--------|
| Stacking | 0.11180 | 3.18379 | 251 (process8) | 2025-12-20 | ❌ Exploded |
| Stacking | 0.11184 | 3.18379 | 8 (meta) | 2025-12-20 | ❌ Exploded |
| Blending | 0.11194 | 6.75855 | 251 (process8) | 2025-12-20 | ❌ Exploded |

**Key Observations:**
- Ensemble models show excellent CV performance (0.11180-0.11194) but catastrophic Kaggle scores
- **Root Cause**: Numerical instability in `expm1()` transformation when predictions are in wrong space
- Meta-model (Lasso, α=0.0005) coefficients: Ridge (0.5826), XGBoost (0.1999), CatBoost (0.1585)
- Blending weights: Ridge (0.565), LightGBM (0.1835), CatBoost (0.1454)
- **Critical Issue**: Predictions explode to 1.5e17 (blending) and 1.5e60 (stacking) instead of ~$178k

### 4.3 Optimal Hyperparameters (Best Models)

#### 4.3.1 CatBoost (Best Kaggle Score: 0.12973)

**Configuration:**
```python
{
    "depth": 5,
    "iterations": 352,
    "learning_rate": 0.048077607907113865,
    "l2_leaf_reg": 6,
    "bagging_temperature": 1,
    "random_strength": 1,
    "task_type": "GPU",
    "devices": "0",
    "loss_function": "RMSE",
    "random_seed": 42
}
```

**Analysis:**
- Moderate depth (5) prevents overfitting while capturing interactions
- Low learning rate (0.048) with moderate iterations (352) balances speed and performance
- L2 regularization (6) provides good generalization
- GPU acceleration enables practical optimization

#### 4.3.2 XGBoost (Best CV RMSE: 0.11436)

**Configuration:**
```python
{
    "n_estimators": 1176,
    "learning_rate": 0.028302545249253366,
    "max_depth": 3,
    "min_child_weight": 2,
    "subsample": 0.6017482152742075,
    "colsample_bytree": 0.7117875626046593,
    "gamma": 0,
    "device": "cuda",
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "random_state": 42
}
```

**Analysis:**
- Shallow trees (depth=3) with many estimators (1176) for ensemble strength
- Aggressive subsampling (0.60) and column sampling (0.71) prevents overfitting
- Very low learning rate (0.028) requires more trees but improves generalization

#### 4.3.3 LightGBM (Best CV RMSE: 0.11795)

**Configuration:**
```python
{
    "n_estimators": 583,
    "learning_rate": 0.06851098495520465,
    "max_depth": 3,
    "num_leaves": 27,
    "min_child_samples": 5,
    "subsample": 0.9225951476283494,
    "colsample_bytree": 0.6291735506636706,
    "reg_alpha": 0,
    "reg_lambda": 9,
    "device_type": "gpu",
    "objective": "regression",
    "random_state": 42
}
```

**Analysis:**
- Leaf-wise growth with moderate num_leaves (27) for efficiency
- Higher learning rate (0.069) than XGBoost, fewer trees needed
- L2 regularization (reg_lambda=9) provides good generalization

#### 4.3.4 Random Forest (Best: 0.12749)

**Configuration:**
```python
{
    "n_estimators": 823,
    "max_depth": 17,
    "max_features": 0.3406184975697621,
    "min_samples_leaf": 2,
    "min_samples_split": 6,
    "criterion": "squared_error",
    "random_state": 42
}
```

**Analysis:**
- Deep trees (depth=17) with moderate feature sampling (0.34)
- Balanced leaf/split constraints prevent overfitting

#### 4.3.5 SVR (Best: 0.14561)

**Configuration:**
```python
{
    "C": 9.736134734118966,
    "gamma": 0.00010012301898827114,
    "epsilon": 0.004069359462567132,
    "kernel": "rbf"
}
```

**Analysis:**
- Moderate C (9.74) balances margin and errors
- Very small gamma (0.0001) creates smooth decision boundaries
- Tight epsilon (0.004) for precise regression

### 4.4 Model Comparison Analysis

#### 4.4.1 Prediction Distribution Analysis

From ModelComparison.ipynb analysis:

**Valid Models** (predictions in reasonable range: $10k-$2M):
- Tree-based models: XGBoost, LightGBM, CatBoost, Random Forest
- Linear models: Ridge, Lasso, Elastic Net
- Kernel: SVR

**Invalid Models** (exploded predictions):
- Blending: Mean ~1.5e17 (should be ~$178k)
- Stacking: Mean ~1.5e60 (should be ~$178k)

**Distribution Statistics:**
- Training target: Mean $180,921, Std $79,443, Range $34,900-$755,000
- Best models (CatBoost/XGBoost) closely match training distribution
- Exploded ensembles show extreme outliers

#### 4.4.2 Correlation Analysis

**Highly Correlated Model Pairs** (correlation > 0.95):
- XGBoost ↔ LightGBM: 0.98
- XGBoost ↔ CatBoost: 0.97
- LightGBM ↔ CatBoost: 0.96

**Implications:**
- Tree-based models make similar predictions (low ensemble diversity)
- Linear models show different patterns (higher diversity)
- Ensemble methods benefit from diverse base models

#### 4.4.3 CV vs Kaggle Score Gap Analysis

| Model | CV RMSE | Kaggle RMSLE | Gap | Interpretation |
|-------|---------|--------------|-----|----------------|
| CatBoost | 0.12017 | 0.12973 | 0.00956 | ✅ Excellent (minimal overfitting) |
| XGBoost | 0.11436 | 0.13094 | 0.01658 | ✅ Good (slight overfitting) |
| LightGBM | 0.11795 | - | - | ✅ Good (no submission) |
| Ridge | 0.09665 | 1.41358 | 1.31693 | ❌ Severe overfitting |
| Random Forest | 0.12749 | - | - | ✅ Good (no submission) |

**Key Findings:**
- Tree-based models show small CV-Kaggle gaps (<0.02), indicating good generalization
- Ridge shows catastrophic overfitting (gap >1.3), likely due to linear model limitations
- Best models balance CV performance with generalization

### 4.5 Feature Engineering Impact

#### 4.5.1 Process6 vs Process8 Comparison

| Model | Process6 (264 features) | Process8 (248-251 features) | Change |
|-------|-------------------------|----------------------------|--------|
| CatBoost | 0.12973 (Kaggle) | 0.13081 (Kaggle) | +0.00108 |
| XGBoost | 0.13335 (Kaggle) | 0.13094 (Kaggle) | -0.00241 ✅ |
| Ridge | 1.41358 (Kaggle) | - | - |

**Analysis:**
- Process8 (with target encoding) shows mixed results
- XGBoost improved with process8, suggesting better feature representation
- CatBoost slightly worse, possibly due to fewer features or target encoding noise
- Feature selection (264→248) removes noise without significant performance loss

#### 4.5.2 Feature Count Evolution

- **Process1-5**: ~80 raw features
- **Process6**: ~264 features (after one-hot encoding)
- **Process7**: ~248-251 features (after feature selection)
- **Process8**: ~248-251 features (after target encoding, some categoricals removed)

**Impact:**
- Feature selection reduces noise (264→248) with minimal performance impact
- Target encoding adds informative features but removes original categoricals
- Optimal feature count: 248-251 (balance between information and overfitting risk)

### 4.6 Key Findings & Insights

1. **Tree-based models dominate**: CatBoost, XGBoost, and LightGBM significantly outperform linear models (0.01-0.02 RMSLE improvement)

2. **Log transformation is critical**: Without transformation, models struggle with skewed distribution; training in log space aligns with RMSLE metric

3. **Feature engineering matters**: 
   - Age features, aggregates, interactions provide substantial improvements
   - Target encoding adds value but requires careful implementation
   - Feature selection removes noise without significant loss

4. **Hyperparameter optimization essential**: Optuna optimization improved XGBoost from 0.13335 to 0.11436 CV RMSE (17% improvement)

5. **Ensemble challenges**: Blending and stacking show excellent CV performance but catastrophic Kaggle scores due to numerical instability

6. **GPU acceleration enables deeper search**: Allows practical 20+ trial Optuna optimization with 5-fold CV

7. **Overfitting detection**: CV-Kaggle gap is critical metric; tree-based models show <0.02 gap, linear models show >1.0 gap

8. **Model diversity**: Tree-based models highly correlated (>0.95), limiting ensemble benefits; need more diverse base models

### 4.7 Best Model Summary

**Winner: CatBoost** with RMSLE 0.12973 (best score as of 2025-12-19)

**Why CatBoost Won:**
- Superior categorical feature handling (native support)
- Best balance of CV performance (0.12017) and generalization (gap: 0.00956)
- Optimal hyperparameters found through Optuna (depth=5, lr=0.048, iterations=352)
- GPU acceleration enabled practical optimization

**Hyperparameters:**
- Depth: 5 (moderate complexity)
- Iterations: 352 (efficient training)
- Learning rate: 0.048 (balanced convergence)
- L2 regularization: 6 (good generalization)
- GPU acceleration: Enabled

**Performance:**
- CV RMSE: 0.12017
- Kaggle RMSLE: 0.12973
- Features: 264 (process6, best submission)
- Runtime: ~20 minutes (Optuna optimization)

---

## 5. Technical Implementation

### 5.1 Architecture

- **Language**: Python 3.8+
- **ML Libraries**: scikit-learn, XGBoost, LightGBM, CatBoost, Optuna
- **GPU Acceleration**: CUDA for tree-based models
- **Automation**: Sequential execution, automatic hyperparameter optimization
- **Kaggle Integration**: Automatic score retrieval, duplicate submission prevention
- **Validation**: Comprehensive sanity checks at every stage

### 5.2 Pipeline Automation

- **Preprocessing**: Automated 8-stage pipeline execution (`notebooks/preprocessing/run_preprocessing.py`)
- **Model Training**: Sequential execution with `scripts/run_all_models_parallel.py`
- **Hyperparameter Optimization**: Optuna with TPE sampler
- **Submission Management**: Automatic score retrieval and logging (`utils/kaggle_helper.py`)
- **Model Comparison**: Automated comparison tools (`scripts/compare_models.py`, `scripts/quick_model_comparison.py`)

### 5.3 Reproducibility

- Fixed random seeds (42) across all models
- Consistent cross-validation strategy (5-fold KFold)
- Version-controlled configuration files (`config_local/`)
- Comprehensive logging of hyperparameters and results (`runs/model_performance.csv`)
- Feature engineering tracking (`runs/feature_definitions/`)

### 5.4 Main Running Files

**Preprocessing**:
- `notebooks/preprocessing/run_preprocessing.py` - Main preprocessing orchestrator
- `notebooks/preprocessing/1cleaning.py` - Stage 1
- `notebooks/preprocessing/2dataEngineering.py` - Stage 2
- `notebooks/preprocessing/3skewKurtosis.py` - Stage 3
- `notebooks/preprocessing/4featureEngineering.py` - Stage 4
- `notebooks/preprocessing/5scaling.py` - Stage 5
- `notebooks/preprocessing/6categorialEncode.py` - Stage 6
- `notebooks/preprocessing/7featureSelection.py` - Stage 7
- `notebooks/preprocessing/8targetEncoding.py` - Stage 8

**Model Training**:
- `scripts/run_all_models_parallel.py` - Main model training orchestrator
- `notebooks/Models/*.py` - Individual model scripts (0-11)

**Kaggle Submission**:
- `scripts/submit_model.py` - Generalized submission script
- `scripts/submit_all_models.py` - Submit all models interactively
- `scripts/get_kaggle_score.py` - Retrieve and log Kaggle scores
- `scripts/check_submission_status.py` - View submission status

**Comparison & Analysis**:
- `scripts/compare_models.py` - Generate visual comparison plots
- `scripts/quick_model_comparison.py` - Text-based comparison report
- `scripts/run_model_comparison.py` - Execute comparison analysis
- `notebooks/ModelComparison.ipynb` - Interactive Jupyter notebook for model comparison
- `scripts/analyze_best_model.py` - Identify best-performing models

**Utilities**:
- `utils/data.py` - Data loading and saving utilities
- `utils/models.py` - Model saving/loading utilities
- `utils/metrics.py` - Performance logging (`log_model_result`, `log_kaggle_score`)
- `utils/optimization.py` - Optuna optimization wrapper
- `utils/kaggle_helper.py` - Kaggle API integration
- `utils/model_wrapper.py` - Validation wrappers
- `utils/checks.py` - Comprehensive sanity checks
- `utils/model_validation.py` - Model-specific validation

---

## 6. Model Comparison & Analysis

### 6.1 Comparison Tools

**Visual Comparisons** (`scripts/compare_models.py`):
- Correlation heatmap between model predictions
- Distribution comparison (KDE plots) vs. training target
- Boxplot comparison showing ranges and outliers
- Pairwise scatter plots for top models
- Output saved to: `runs/latest/comparison/`

**Text-Based Comparison** (`scripts/quick_model_comparison.py`):
- Statistical summary (mean, median, std, min, max)
- Performance metrics (CV RMSE, Kaggle score)
- Model rankings by CV RMSE and Kaggle score
- Highly correlated model pairs identification

**Interactive Notebook** (`notebooks/ModelComparison.ipynb`):
- Dynamic loading of predictions and performance metrics
- Comprehensive statistics table
- Model rankings
- Correlation analysis
- Distribution and boxplot visualizations
- Always up-to-date with latest results

### 6.2 Current Comparison Results

**Valid Models** (predictions in reasonable range):
- XGBoost, LightGBM, CatBoost, Random Forest, SVR, Ridge, Lasso, Elastic Net

**Invalid Models** (exploded predictions):
- Blending: Mean prediction ~1.5e17 (should be ~$178k)
- Stacking: Mean prediction ~1.5e60 (should be ~$178k)

**Model Correlations**:
- XGBoost, LightGBM, CatBoost: Very high correlation (>0.95)
- Tree-based models show similar prediction patterns
- Linear models (Ridge, Lasso) show different patterns

---

## 7. Insights & Conclusions

### 7.1 Critical Success Factors

1. **Target Transformation**: Log transformation is the single most important preprocessing step
2. **Feature Engineering**: Domain knowledge (age, aggregates, interactions) significantly improves predictions
3. **Feature Selection**: Removing noise features improves model performance
4. **Target Encoding**: Cross-validated target encoding captures category-target relationships effectively
5. **Model Selection**: Tree-based models capture non-linear relationships better than linear models
6. **Hyperparameter Optimization**: Systematic search (Optuna) finds better configurations than manual tuning
7. **Computational Resources**: GPU acceleration enables practical deep hyperparameter search
8. **Validation**: Comprehensive sanity checks prevent errors and ensure data integrity

### 7.2 Known Issues & Future Work

**Current Issues**:
- **Blending Model**: Produces exploded predictions (likely space mismatch issue)
- **Stacking Model**: Produces exploded predictions (likely numerical instability in meta-model)
- **Fix Needed**: Ensure consistent space (log vs. real) in ensemble methods, add bounds checking

**Future Work**:
- Fix blending and stacking numerical issues
- Explore more domain-specific features (neighborhood effects, market trends)
- Improve model interpretability (SHAP values, feature importance analysis)
- Optimize computational cost (faster feature selection, parallel preprocessing)
- Experiment with different ensemble methods (voting, weighted voting)

### 7.3 Broader Implications

This work demonstrates:
- The importance of proper data preprocessing in real-world ML applications
- The value of systematic hyperparameter optimization
- The effectiveness of tree-based models for non-linear regression
- The critical role of validation and sanity checks
- The challenges of ensemble methods (numerical stability, space consistency)
- The practical benefits of GPU acceleration in model development

---

## 8. Project Structure

```
house-prices-starter/
├── docs/                    # Documentation
│   ├── FLAGSHIP_LOG.md      # This file - project showcase
│   ├── TECHNICAL_LOG.md     # Detailed technical documentation
│   └── ...                  # Additional documentation
├── notebooks/
│   ├── preprocessing/       # 8-stage preprocessing pipeline
│   │   ├── run_preprocessing.py  # Main orchestrator
│   │   ├── 1cleaning.py
│   │   ├── 2dataEngineering.py
│   │   ├── 3skewKurtosis.py
│   │   ├── 4featureEngineering.py
│   │   ├── 5scaling.py
│   │   ├── 6categorialEncode.py
│   │   ├── 7featureSelection.py
│   │   └── 8targetEncoding.py
│   ├── Models/              # 12 model implementations (0-11)
│   │   ├── 0linearRegression.py
│   │   ├── 1linearRegUpdated.py
│   │   ├── 2ridgeModel.py
│   │   ├── 3lassoModel.py
│   │   ├── 4elasticNetModel.py
│   │   ├── 5randomForestModel.py
│   │   ├── 6svrModel.py
│   │   ├── 7XGBoostModel.py
│   │   ├── 8lightGbmModel.py
│   │   ├── 9catBoostModel.py
│   │   ├── 10blendingModel.py
│   │   └── 11stackingModel.py
│   ├── ModelComparison.ipynb  # Interactive comparison notebook
│   └── Journal.ipynb        # Project journal
├── scripts/                 # Automation & utility scripts
│   ├── run_all_models_parallel.py  # Main model training script
│   ├── submit_model.py      # Generalized submission script
│   ├── submit_all_models.py # Interactive submission
│   ├── compare_models.py   # Visual comparison
│   ├── quick_model_comparison.py  # Text comparison
│   ├── run_model_comparison.py    # Comparison executor
│   ├── analyze_best_model.py      # Best model analysis
│   ├── get_kaggle_score.py  # Score retrieval
│   └── check_submission_status.py # Submission status
├── config_local/            # Configuration & hyperparameters
│   ├── local_config.py      # Paths and directories
│   └── model_config.py      # Model configurations
├── utils/                   # Utilities
│   ├── data.py              # Data loading/saving
│   ├── models.py            # Model utilities
│   ├── metrics.py           # Performance logging
│   ├── optimization.py      # Optuna wrapper
│   ├── kaggle_helper.py     # Kaggle API integration
│   ├── model_wrapper.py     # Validation wrappers
│   ├── checks.py            # Sanity checks
│   ├── model_validation.py  # Model validation
│   └── engineering.py      # Feature engineering utilities
├── data/
│   ├── raw/                 # Original Kaggle data
│   ├── interim/             # Processed data (8 stages)
│   │   ├── train/           # Training data (process1-8)
│   │   ├── test/            # Test data (process1-8)
│   │   └── oof/             # Out-of-fold predictions
│   ├── processed/           # Feature summaries
│   └── submissions/         # Kaggle submission files
└── runs/                    # Results and outputs
    ├── model_performance.csv  # Performance log
    ├── latest/
    │   └── comparison/      # Comparison plots
    └── feature_definitions/  # Feature engineering tracking
```

---

## 9. Quick Reference

### Setup
```bash
# Windows
.\scripts\setup\setup_venv.ps1

# Linux/Mac
./scripts/setup/setup_venv.sh
```

### Execution

**Preprocessing**:
```bash
python notebooks/preprocessing/run_preprocessing.py
```

**Train Models**:
```bash
python scripts/run_all_models_parallel.py
```

**Submit to Kaggle**:
```bash
python scripts/submit_model.py <model_name>
# Example: python scripts/submit_model.py catboost
```

**Compare Models**:
```bash
# Visual comparison
python scripts/compare_models.py

# Text comparison
python scripts/quick_model_comparison.py

# Interactive notebook
jupyter notebook notebooks/ModelComparison.ipynb
```

**Check Status**:
```bash
# Submission status
python scripts/check_submission_status.py

# Best models
python scripts/analyze_best_model.py
```

### Best Model
- **CatBoost** with RMSLE 0.12973 (best score as of 2025-12-19)
- Hyperparameters: depth=5, iterations=352, learning_rate=0.048
- Optimized via Optuna with 5-fold CV
- Features: 248-251 (process8, varies by feature selection)

---

## References

- **Competition**: [Kaggle House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **Best Score**: RMSLE 0.12973 (CatBoost)
- **Date**: December 2025
- **Status**: Active development, ensemble methods need fixes

---

*This logbook serves as the flagship showcase of the project. For detailed technical documentation, implementation details, and error logs, see `TECHNICAL_LOG.md`.*
