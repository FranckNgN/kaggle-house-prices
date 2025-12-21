# 🚀 Running CatBoost on Kaggle GPU - Quick Instructions

## ✅ Pre-flight Checklist

Before you start, make sure:

1. **Code is pushed to GitHub**:
   ```bash
   git push origin master
   ```

2. **You have Kaggle account** with GPU quota available (30 hours/week free tier)

3. **Preprocessed data exists** (or the notebook will need to run preprocessing first)

## 🎯 Quick Steps

### 1. Push Latest Code (if not done)
```bash
# Check status
git status

# Push if needed
git push origin master
```

### 2. Create Kaggle Notebook

1. Go to: https://www.kaggle.com/code
2. Click **"New Notebook"**
3. In Settings (right panel):
   - **Accelerator**: Select **"GPU"** (P100 or T4)
   - **Internet**: Enable ✅

### 3. Add Competition Dataset

1. Click **"Add Data"** (right panel)
2. Search: `house-prices-advanced-regression-techniques`
3. Click **"Add"**

### 4. Use the Notebook

**Option A: Upload the notebook file**
- File → Upload Notebook
- Select: `kaggle/notebooks/run_catboost_kaggle.ipynb`

**Option B: Copy cells manually**
- Copy cells from `kaggle/notebooks/run_catboost_kaggle.ipynb`
- Paste into Kaggle notebook

### 5. Run All Cells

Click **"Run All"** or run cells sequentially. The notebook will:

1. ✅ Clone repository from GitHub
2. ✅ Install dependencies
3. ✅ Setup Kaggle environment (symlinks, paths)
4. ✅ Verify GPU availability
5. ✅ Train CatBoost model with GPU acceleration
6. ✅ Generate submission file

### 6. Download Results

After training completes:

1. Go to **"Data"** panel (right side)
2. Navigate to: `/kaggle/working/project/data/submissions/`
3. Download the submission CSV file
4. Submit on competition page

## 📊 What You'll See

### Expected Output:
```
[OK] Repository cloned to /kaggle/working/project
[OK] Dependencies installed
[OK] GPU available: Tesla P100
[INFO] GPU available. catboost will use GPU.
...
CATBOOST MODEL TRAINING
Training samples: 1460
Features: XXX
Optuna trials: XXX
CV folds: 5
Device: GPU (accelerated)
...
CATBOOST MODEL COMPLETE!
Total runtime: XX.X minutes
Submission saved: /kaggle/working/project/data/submissions/...
Best CV RMSE: 0.XXXXXX
```

## 🔧 Troubleshooting

### GPU Not Detected
- ✅ Restart notebook session after enabling GPU
- ✅ Check Settings → Accelerator = "GPU"

### Import Errors
- ✅ Check Cell 1 output - dependencies should install
- ✅ Verify repository cloned correctly

### Data Not Found
- ✅ Ensure competition dataset is added
- ✅ Check Cell 2 output - symlinks should be created

## 📝 Files Generated

After successful run:
- **Submission**: `data/submissions/catboost_YYYYMMDD_HHMMSS.csv`
- **Performance Log**: `runs/model_performance.csv`
- **Model Info**: `runs/feature_definitions/catboost_*.json`

## 🎓 Next Steps

1. Download submission file
2. Submit to competition: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/submit
3. Compare with local results
4. Iterate and improve!

---

For detailed documentation, see:
- `KAGGLE_CATBOOST_QUICKSTART.md` - Detailed guide
- `docs/HYBRID_WORKFLOW.md` - Complete workflow documentation

