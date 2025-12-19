# House Prices - Advanced Regression Techniques

Kaggle competition project for predicting house sale prices using a structured preprocessing pipeline and ensemble modeling approach.

## 🎯 Project Overview

This project implements a comprehensive 6-stage preprocessing pipeline followed by multiple machine learning models, with final predictions generated through model blending. The best performing model achieved **Rank 1215** on the Kaggle leaderboard with **RMSLE 0.12647**.

## 📋 Features

- **6-Stage Preprocessing Pipeline**: Cleaning → Data Engineering → Skew Normalization → Feature Engineering → Scaling → Encoding
- **7 ML Models**: Linear Regression, Ridge, Lasso, ElasticNet, XGBoost, LightGBM, CatBoost
- **Ensemble Blending**: Weighted combination of top tree-based models
- **Automated Pipeline**: Script-based execution of preprocessing notebooks
- **GPU Acceleration**: Tree-based models utilize CUDA for faster training

## 🚀 Quick Start

### Installation

**Windows (PowerShell):**
```powershell
# Clone the repository
git clone https://github.com/YOUR_USERNAME/kaggle-house-prices.git
cd kaggle-house-prices

# Setup virtual environment (one-time)
.\setup_venv.ps1
```

**Linux/Mac:**
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/kaggle-house-prices.git
cd kaggle-house-prices

# Setup virtual environment (one-time)
chmod +x setup_venv.sh
./setup_venv.sh
```

**Note:** All Python scripts automatically use the virtual environment when available. You don't need to manually activate it.

### Configuration

1. Copy and configure local paths:
```bash
cp config_local/local_config.py.example config_local/local_config.py
# Edit config_local/local_config.py with your data paths
```

2. Place Kaggle data files in `data/raw/`:
   - `train.csv`
   - `test.csv`
   - `data_description.txt`

### Run Preprocessing Pipeline

```bash
# From project root
python notebooks/preprocessing/run_preprocessing.py
```

This will:
1. Clean the interim directory
2. Execute notebooks 1-6 sequentially
3. Generate processed datasets in `data/interim/`

### Train Models

Run individual model notebooks in `notebooks/Models/` or use the blending script:

```bash
python notebooks/Models/8blendingModel.py
```

## 📁 Project Structure

```
house-prices-starter/
├── data/
│   ├── raw/              # Original Kaggle data
│   ├── interim/          # Intermediate processed files (process1-6)
│   ├── processed/        # Final processed features
│   └── submissions/      # Kaggle submission files
├── notebooks/
│   ├── preprocessing/    # 6-stage preprocessing pipeline
│   └── Models/           # Model training notebooks
├── config_local/         # Configuration management
├── utils/                # Utility functions
└── tests/                # Integration tests
```

## 🔧 Preprocessing Pipeline

1. **Cleaning**: Missing value imputation (numeric → 0, categorical → "<None>")
2. **Data Engineering**: Target transformation (log1p), outlier removal
3. **Skew Normalization**: Yeo-Johnson transformation for skewed features
4. **Feature Engineering**: Age features, K-means clustering
5. **Scaling**: StandardScaler for continuous numeric features
6. **Encoding**: One-hot encoding for categorical variables

## 📊 Model Performance

| Model | Kaggle RMSLE | Rank | CV RMSE |
|-------|--------------|------|---------|
| LightGBM | 0.12647 | 1215 | - |
| XGBoost | 0.13125 | 1215 | 0.1155 |
| CatBoost | - | - | 0.12054 |
| Ridge | 0.13272 | 2167 | 20,562.79 |
| ElasticNet | 0.13677 | 2567 | - |
| Lasso | 0.14850 | - | 24,120.01 |
| Linear Regression | 0.15051 | 4315 | ~23,240 |

## 🛠️ Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

The project follows Python best practices:
- Type hints in Python modules
- Docstrings for functions
- Consistent code formatting
- Error handling and validation

## 📝 Key Technical Decisions

- **Target Transformation**: `log1p(SalePrice)` for better distribution properties
- **Outlier Removal**: Domain-specific (GrLivArea > 4000 & SalePrice < 300000)
- **Cross-Validation**: 5-fold KFold with consistent random_state=42
- **Ensemble Weights**: XGBoost (2.0), LightGBM (0.5), CatBoost (1.0)

## 📚 Documentation

See [Ship_Log.md](Ship_Log.md) for detailed project documentation.

## 🤝 Contributing

Contributions welcome! Please ensure:
- Code follows project style guidelines
- Tests pass
- Documentation is updated

## 📄 License

This project is for educational purposes as part of the Kaggle competition.

## 🙏 Acknowledgments

- Kaggle for hosting the competition
- Open source ML libraries (scikit-learn, XGBoost, LightGBM, CatBoost)

