# House Prices Prediction - Flagship Log

**A Comprehensive Machine Learning Pipeline for Real Estate Price Prediction**

*Project Journal & Showcase*

---

## Abstract

This project implements a complete machine learning pipeline for predicting house sale prices using advanced regression techniques. Through systematic preprocessing, feature engineering, and ensemble modeling, we achieve **RMSLE 0.12973** (CatBoost) on the Kaggle leaderboard. The work demonstrates the critical importance of target transformation, feature engineering, and model selection in regression tasks.

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

A 6-stage systematic preprocessing approach:

1. **Cleaning**: Handle missing values (numeric→0, categorical→"<None>")
2. **Data Engineering**: Log-transform target, remove domain-specific outliers (2 removed)
3. **Skew Normalization**: Yeo-Johnson transformation for features with |skew| > 0.75
4. **Feature Engineering**: 
   - Temporal features (age, garage age, remodel age)
   - Aggregate features (total square footage, total bathrooms)
   - Interaction features (quality × size, condition × age)
   - K-means clustering (k=4) on key features
5. **Scaling**: StandardScaler for continuous numeric features only
6. **Encoding**: One-hot encoding for categorical features

**Result**: 264 features ready for modeling

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
$$\hat{y}_{blend} = \frac{2.0 \cdot \hat{y}_{XGB} + 0.5 \cdot \hat{y}_{LGB} + 1.0 \cdot \hat{y}_{CAT}}{3.5}$$

**Stacking** (Meta-learner):
$$\hat{y}_{stack} = g(\hat{y}_{XGB}, \hat{y}_{LGB}, \hat{y}_{CAT}, \hat{y}_{Ridge}, \ldots)$$

where $g$ is a meta-model (Lasso with α=0.0005).

### 3.3 Optimization Strategy

**Hyperparameter Search**: Bayesian optimization using Optuna (TPE sampler)
$$\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta} \in \Theta} \mathbb{E}[L_{CV}(\boldsymbol{\theta})]$$

**Cross-Validation**: 5-fold KFold for robust performance estimation
$$\text{CV-RMSE} = \frac{1}{5}\sum_{k=1}^{5} \text{RMSE}_k$$

**Runtime Optimization**: Target ~20 minutes per model with balanced search depth

---

## 4. Results & Performance

### 4.1 Model Performance Summary

| Model | Kaggle RMSLE | Type | Key Feature |
|-------|--------------|------|-------------|
| **CatBoost** | **0.12973** | Tree-based | Categorical handling |
| XGBoost | 0.13335 | Tree-based | GPU acceleration |
| Random Forest | 0.14460 | Tree-based | Ensemble robustness |
| Blending | 0.14480 | Ensemble | Weighted combination |
| SVR | 0.18191 | Kernel | Non-linear mapping |
| Elastic Net | 0.63422 | Linear | Feature selection |
| Stacking | 0.62673 | Ensemble | Meta-learning |
| Ridge | 1.41358 | Linear | L2 regularization |
| Lasso | 1.97336 | Linear | L1 regularization |

### 4.2 Key Findings

1. **Tree-based models dominate**: CatBoost, XGBoost, and LightGBM significantly outperform linear models
2. **Log transformation is critical**: Without transformation, models struggle with skewed distribution
3. **Feature engineering matters**: Age features, aggregates, and interactions provide substantial improvements
4. **Ensemble benefits are marginal**: Best single model (CatBoost) performs comparably to ensembles
5. **GPU acceleration enables deeper search**: Allows more thorough hyperparameter optimization

---

## 5. Technical Implementation

### 5.1 Architecture

- **Language**: Python 3.8+
- **ML Libraries**: scikit-learn, XGBoost, LightGBM, CatBoost, Optuna
- **GPU Acceleration**: CUDA for tree-based models
- **Automation**: Parallel execution, automatic hyperparameter optimization
- **Kaggle Integration**: Automatic score retrieval, duplicate submission prevention

### 5.2 Pipeline Automation

- **Preprocessing**: Automated 6-stage pipeline execution
- **Model Training**: Parallel execution with ProcessPoolExecutor
- **Hyperparameter Optimization**: Optuna with TPE sampler
- **Submission Management**: Automatic score retrieval and logging

### 5.3 Reproducibility

- Fixed random seeds (42) across all models
- Consistent cross-validation strategy (5-fold KFold)
- Version-controlled configuration files
- Comprehensive logging of hyperparameters and results

---

## 6. Insights & Conclusions

### 6.1 Critical Success Factors

1. **Target Transformation**: Log transformation is the single most important preprocessing step
2. **Feature Engineering**: Domain knowledge (age, aggregates, interactions) significantly improves predictions
3. **Model Selection**: Tree-based models capture non-linear relationships better than linear models
4. **Hyperparameter Optimization**: Systematic search (Optuna) finds better configurations than manual tuning
5. **Computational Resources**: GPU acceleration enables practical deep hyperparameter search

### 6.2 Limitations & Future Work

- **Ensemble Complexity**: Stacking underperformed, suggesting overfitting or insufficient base model diversity
- **Feature Engineering**: Could explore more domain-specific features (neighborhood effects, market trends)
- **Model Interpretability**: Tree-based models are less interpretable than linear models
- **Computational Cost**: Deep optimization requires significant time and resources

### 6.3 Broader Implications

This work demonstrates:
- The importance of proper data preprocessing in real-world ML applications
- The value of systematic hyperparameter optimization
- The effectiveness of ensemble methods when base models are diverse
- The practical benefits of GPU acceleration in model development

---

## 7. Quick Start

### Setup
```bash
# Windows
.\scripts\setup\setup_venv.ps1

# Linux/Mac
./scripts/setup/setup_venv.sh
```

### Execution
```bash
# Preprocess
python notebooks/preprocessing/run_preprocessing.py

# Train models
python scripts/run_all_models_parallel.py

# Submit to Kaggle
python scripts/submit_to_kaggle.py data/submissions/model.csv
```

### Best Model
- **CatBoost** with RMSLE 0.12973
- Hyperparameters: depth=5, iterations=352, learning_rate=0.048
- Optimized via Optuna with 5-fold CV

---

## 📚 Additional Documentation

- **[Technical Log](docs/TECHNICAL_LOG.md)** - Detailed technical documentation, errors, implementation details
- **[Venv Usage](docs/VENV_USAGE.md)** - Virtual environment setup guide
- **[Scripts Guide](scripts/README.md)** - Script usage and automation

---

## References

- **Competition**: [Kaggle House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **Best Score**: RMSLE 0.12973 (CatBoost)
- **Date**: December 2025

---

*This logbook serves as the flagship showcase of the project. For detailed technical documentation, see [Technical Log](docs/TECHNICAL_LOG.md).*
