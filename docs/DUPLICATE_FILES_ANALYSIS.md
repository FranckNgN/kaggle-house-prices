# Duplicate Files Analysis

## Summary

Found several files with overlapping or duplicate functionality that can be consolidated.

## 1. Submission Scripts (3 files) - HIGH OVERLAP

### Files:
- `scripts/submit_all_models.py` (224 lines) - Interactive menu
- `scripts/submit_all_models_auto.py` (221 lines) - Automatic submission
- `scripts/submit_to_kaggle.py` (60 lines) - Single model submission

### Analysis:
- **`submit_all_models.py`**: Interactive menu that can submit single or all models
- **`submit_all_models_auto.py`**: Non-interactive version that automatically submits all models
- **`submit_to_kaggle.py`**: Command-line tool to submit a single model

### Overlap:
- All three use the same `get_available_submissions()` function (duplicated code)
- `submit_all_models.py` can already do everything the other two do
- `submit_to_kaggle.py` is redundant - `submit_all_models.py` option 1 does the same

### Recommendation:
**Keep**: `submit_all_models.py` (most comprehensive)
**Remove**: 
- `submit_all_models_auto.py` (can be replaced with a flag in `submit_all_models.py`)
- `submit_to_kaggle.py` (functionality already in `submit_all_models.py`)

**Alternative**: Add `--auto` flag to `submit_all_models.py` to make it non-interactive

---

## 2. Comparison Scripts (2 files) - DIFFERENT PURPOSES

### Files:
- `scripts/compare_models.py` (131 lines) - Generates visual plots
- `scripts/quick_model_comparison.py` (148 lines) - Generates text report

### Analysis:
- **`compare_models.py`**: Creates visualizations (heatmaps, distributions, boxplots)
- **`quick_model_comparison.py`**: Creates text-based comparison report with statistics

### Overlap:
- Both use `load_all_submissions()` from `compare_models.py`
- Both compare the same data, just different output formats

### Recommendation:
**Keep both** - They serve different purposes (visual vs text output)
**Consider**: Moving `load_all_submissions()` to a shared utility module

---

## 3. Status/Check Scripts (3 files) - DIFFERENT PURPOSES

### Files:
- `scripts/check_model_progress.py` (128 lines) - Check running training processes
- `scripts/check_submission_status.py` (153 lines) - Check Kaggle submission status
- `scripts/get_kaggle_score.py` (61 lines) - Get latest Kaggle score

### Analysis:
- **`check_model_progress.py`**: Monitors running Python processes (training)
- **`check_submission_status.py`**: Shows which models have been submitted to Kaggle
- **`get_kaggle_score.py`**: Fetches and logs a single Kaggle score

### Overlap:
- `check_submission_status.py` and `get_kaggle_score.py` both interact with Kaggle API
- Some functionality overlap in checking submission status

### Recommendation:
**Keep all** - They serve different purposes:
- `check_model_progress.py` - Training monitoring
- `check_submission_status.py` - Submission tracking
- `get_kaggle_score.py` - Score retrieval

**Consider**: Could merge `get_kaggle_score.py` into `check_submission_status.py` as an option

---

## 4. Performance Scripts (1 file)

### Files:
- `scripts/show_performance.py` (104 lines)

### Analysis:
- Unique functionality - shows performance logs from CSV
- No duplicates found

### Recommendation:
**Keep** - No duplicates

---

## Recommended Actions

### High Priority (Remove Duplicates):
1. **Remove `submit_all_models_auto.py`**
   - Add `--auto` flag to `submit_all_models.py` instead
   - Or keep but document that it's a convenience wrapper

2. **Remove `submit_to_kaggle.py`**
   - Functionality already in `submit_all_models.py` (option 1)

### Medium Priority (Consolidate Code):
3. **Extract `get_available_submissions()` to shared utility**
   - Currently duplicated in 3 submission scripts
   - Move to `utils/kaggle_helper.py` or create `utils/submission_utils.py`

4. **Consider merging `get_kaggle_score.py` into `check_submission_status.py`**
   - Add `--get-score` option to `check_submission_status.py`

### Low Priority (Documentation):
5. **Update README.md** to clarify which scripts to use
6. **Add deprecation notices** if keeping old scripts for backward compatibility

---

## Files to Remove

1. `scripts/submit_all_models_auto.py` (221 lines) - Redundant
2. `scripts/submit_to_kaggle.py` (60 lines) - Redundant

**Total lines to remove**: ~281 lines

---

## Files to Keep (with modifications)

1. `scripts/submit_all_models.py` - Enhance with `--auto` flag
2. `scripts/compare_models.py` - Keep as-is
3. `scripts/quick_model_comparison.py` - Keep as-is
4. `scripts/check_model_progress.py` - Keep as-is
5. `scripts/check_submission_status.py` - Consider adding score retrieval
6. `scripts/get_kaggle_score.py` - Keep or merge into check_submission_status
7. `scripts/show_performance.py` - Keep as-is

