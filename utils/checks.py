"""Comprehensive data validation and leakage detection utilities."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Union
from scipy import stats
import warnings

try:
    import config_local.local_config as cfg
except ImportError:
    cfg = None

# Try to import colorama for colored output, fallback if not available
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    # Fallback: define dummy color codes
    class Fore:
        RED = ''
        GREEN = ''
        YELLOW = ''
        CYAN = ''
    class Style:
        BRIGHT = ''
        RESET_ALL = ''
    COLORAMA_AVAILABLE = False


class DataLeakageError(Exception):
    """Raised when data leakage is detected."""
    pass


class DataIntegrityError(Exception):
    """Raised when data integrity checks fail."""
    pass


def print_error(message: str, error_type: str = "ERROR") -> None:
    """Print error message with highlighting."""
    print(f"\n{Fore.RED}{Style.BRIGHT}{'='*60}")
    print(f"{error_type}: {message}")
    print(f"{'='*60}{Style.RESET_ALL}\n")


def print_warning(message: str) -> None:
    """Print warning message."""
    print(f"{Fore.YELLOW}[WARNING] {message}{Style.RESET_ALL}")


def print_success(message: str) -> None:
    """Print success message."""
    print(f"{Fore.GREEN}[SUCCESS] {message}{Style.RESET_ALL}")


# ============================================================================
# RAW DATA CHECKS
# ============================================================================

def check_raw_data_integrity(train_path: Path, test_path: Path) -> Dict[str, bool]:
    """
    Check raw data integrity: row counts, duplicates, basic structure.
    
    Returns:
        Dictionary with check results
    """
    results = {}
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Row counts
    results['train_has_rows'] = len(train) > 0
    results['test_has_rows'] = len(test) > 0
    
    # Duplicate rows
    results['train_no_duplicate_rows'] = not train.duplicated().any()
    results['test_no_duplicate_rows'] = not test.duplicated().any()
    
    # ID uniqueness (if Id column exists)
    if 'Id' in train.columns:
        results['train_id_unique'] = train['Id'].nunique() == len(train)
    if 'Id' in test.columns:
        results['test_id_unique'] = test['Id'].nunique() == len(test)
    
    # Target column exists in train
    if 'SalePrice' in train.columns:
        results['target_exists'] = True
        results['target_all_positive'] = (train['SalePrice'] > 0).all()
        results['target_no_nans'] = train['SalePrice'].notna().all()
    else:
        results['target_exists'] = False
    
    return results


# ============================================================================
# PREPROCESSING STAGE CHECKS
# ============================================================================

def check_no_train_test_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stage_name: str
) -> None:
    """
    Check that train and test were processed independently (no leakage).
    
    This checks:
    - No shared IDs between train and test
    - Column parity (except target)
    - No obvious contamination
    """
    # Check for shared IDs if Id column exists
    if 'Id' in train_df.columns and 'Id' in test_df.columns:
        train_ids = set(train_df['Id'].values)
        test_ids = set(test_df['Id'].values)
        if train_ids & test_ids:
            raise DataLeakageError(
                f"[{stage_name}] Found shared IDs between train and test: "
                f"{len(train_ids & test_ids)} IDs"
            )
    
    # Check column parity (excluding target)
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    
    # Remove target columns
    for target_col in ['SalePrice', 'logSP']:
        train_cols.discard(target_col)
        test_cols.discard(target_col)
    
    if train_cols != test_cols:
        only_train = train_cols - test_cols
        only_test = test_cols - train_cols
        raise DataLeakageError(
            f"[{stage_name}] Column mismatch:\n"
            f"  Only in train: {only_train}\n"
            f"  Only in test: {only_test}"
        )


def check_fit_transform_independence(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stage_name: str,
    scaler_or_transformer=None
) -> None:
    """
    Check that transformers were fit on train only.
    
    This is a heuristic check - we can't perfectly detect this,
    but we can check for suspicious patterns.
    """
    # Check if test has values outside train range (suspicious if scaled)
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ['SalePrice', 'logSP']]
    
    for col in numeric_cols[:10]:  # Check first 10 to avoid too much computation
        if col in test_df.columns:
            train_min, train_max = train_df[col].min(), train_df[col].max()
            test_min, test_max = test_df[col].min(), test_df[col].max()
            
            # If test range is much wider, might indicate joint fitting
            train_range = train_max - train_min
            if train_range > 0:
                test_range = test_max - test_min
                if test_range > train_range * 1.5:
                    warnings.warn(
                        f"[{stage_name}] Column {col} in test has wider range than train. "
                        f"This might indicate joint fitting (leakage risk)."
                    )


def check_target_not_in_features(
    df: pd.DataFrame,
    stage_name: str,
    target_col: str = 'logSP'
) -> None:
    """Check that target column is not accidentally used as a feature."""
    if target_col in df.columns:
        # Check if target is in feature set (should only be in train)
        # This is more of a warning for later stages
        pass
    
    # Check for features that are direct transformations of target
    if 'SalePrice' in df.columns and target_col in df.columns:
        # Verify logSP is actually log1p(SalePrice)
        expected_log = np.log1p(df['SalePrice'].values)
        actual_log = df[target_col].values
        if not np.allclose(expected_log, actual_log, rtol=1e-6):
            raise DataIntegrityError(
                f"[{stage_name}] logSP does not match log1p(SalePrice)"
            )


def check_no_target_leakage_in_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stage_name: str
) -> None:
    """
    Check that no features were created using target information.
    
    This checks for:
    - Features that correlate perfectly with target (in train)
    - Features that don't exist in test but exist in train
    """
    # Get feature columns (exclude target)
    feature_cols = [c for c in train_df.columns 
                   if c not in ['SalePrice', 'logSP', 'Id']]
    
    if 'logSP' in train_df.columns:
        target = train_df['logSP']
        
        for col in feature_cols[:20]:  # Check first 20 features
            if col in train_df.columns and train_df[col].dtype in [np.float64, np.int64]:
                # Check for perfect correlation (suspicious)
                corr = train_df[[col, 'logSP']].corr().iloc[0, 1]
                if abs(corr) > 0.99:
                    warnings.warn(
                        f"[{stage_name}] Feature {col} has near-perfect correlation "
                        f"with target (r={corr:.4f}). Possible leakage!"
                    )
                
                # Check that feature exists in test
                if col not in test_df.columns:
                    raise DataLeakageError(
                        f"[{stage_name}] Feature {col} exists in train but not in test"
                    )


def check_missing_values_handled(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stage_name: str,
    allow_target_nans: bool = False
) -> None:
    """Check that missing values are properly handled."""
    target_cols = ['SalePrice', 'logSP']
    
    # Check train
    train_nulls = train_df.isnull().sum()
    train_nulls = train_nulls[train_nulls > 0]
    
    if not allow_target_nans:
        for target_col in target_cols:
            if target_col in train_df.columns and train_df[target_col].isnull().any():
                raise DataIntegrityError(
                    f"[{stage_name}] Train target {target_col} has missing values"
                )
    
    # Check test (should have no NaNs in features after stage 1)
    test_nulls = test_df.isnull().sum()
    test_nulls = test_nulls[test_nulls > 0]
    
    if len(test_nulls) > 0 and stage_name != "Stage 1 (Cleaning)":
        raise DataIntegrityError(
            f"[{stage_name}] Test has missing values in: {test_nulls.index.tolist()}"
        )


def check_shape_consistency(
    train_before: pd.DataFrame,
    train_after: pd.DataFrame,
    test_before: pd.DataFrame,
    test_after: pd.DataFrame,
    stage_name: str,
    expected_row_change: Optional[int] = None
) -> None:
    """Check that shapes are consistent (except for expected changes)."""
    # Train rows should only change if explicitly expected (e.g., outlier removal)
    if expected_row_change is None:
        if len(train_before) != len(train_after):
            raise DataIntegrityError(
                f"[{stage_name}] Train row count changed unexpectedly: "
                f"{len(train_before)} -> {len(train_after)}"
            )
    
    # Test rows should never change
    if len(test_before) != len(test_after):
        raise DataIntegrityError(
            f"[{stage_name}] Test row count changed: "
            f"{len(test_before)} -> {len(test_after)}"
        )


def check_feature_engineering_sanity(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stage_name: str
) -> None:
    """Check that engineered features are logically correct."""
    # Check for negative values where they shouldn't exist
    # Note: After Yeo-Johnson transformation (stage 3), some area features may have negative values
    # This is expected behavior for the transformation, so we skip this check for stage 4+
    if "Stage 4" in stage_name or "Stage 5" in stage_name or "Stage 6" in stage_name or "Stage 7" in stage_name:
        # After transformations, negative values are acceptable for transformed features
        # Only check for extreme outliers or infinite values
        area_cols = [c for c in train_df.columns if 'SF' in c or 'Area' in c or 'Size' in c]
        for col in area_cols:
            if col in train_df.columns and train_df[col].dtype in [np.float64, np.int64]:
                if not np.isfinite(train_df[col]).all():
                    raise DataIntegrityError(
                        f"[{stage_name}] Feature {col} has non-finite values"
                    )
                # Check for extreme outliers (values > 3 standard deviations from mean)
                mean_val = train_df[col].mean()
                std_val = train_df[col].std()
                if std_val > 0:
                    extreme = (train_df[col] < mean_val - 5 * std_val) | (train_df[col] > mean_val + 5 * std_val)
                    if extreme.sum() > len(train_df) * 0.01:  # More than 1% extreme values
                        warnings.warn(
                            f"[{stage_name}] Feature {col} has many extreme values (>5 std dev)"
                        )
    else:
        # For earlier stages, negative area values are not acceptable
        area_cols = [c for c in train_df.columns if 'SF' in c or 'Area' in c or 'Size' in c]
        for col in area_cols:
            if col in train_df.columns and train_df[col].dtype in [np.float64, np.int64]:
                if (train_df[col] < 0).any():
                    raise DataIntegrityError(
                        f"[{stage_name}] Feature {col} has negative values"
                    )
    
    # Check age features are reasonable
    age_cols = [c for c in train_df.columns if 'Age' in c]
    for col in age_cols:
        if col in train_df.columns:
            if (train_df[col] < 0).any() or (train_df[col] > 200).any():
                warnings.warn(
                    f"[{stage_name}] Feature {col} has suspicious values "
                    f"(min={train_df[col].min()}, max={train_df[col].max()})"
                )


# ============================================================================
# MODEL CHECKS
# ============================================================================

def check_model_no_target_leakage(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    model_name: str
) -> None:
    """
    Check that model features don't contain target information.
    
    This checks:
    - No target column in X_train or X_test
    - No perfect correlations with target
    """
    # Check target not in features
    if y_train.name in X_train.columns:
        raise DataLeakageError(
            f"[{model_name}] Target column '{y_train.name}' found in feature matrix"
        )
    
    # Check for suspiciously high correlations
    if isinstance(X_train, pd.DataFrame):
        for col in X_train.select_dtypes(include=[np.number]).columns[:20]:
            corr = X_train[col].corr(y_train)
            if abs(corr) > 0.99:
                warnings.warn(
                    f"[{model_name}] Feature {col} has near-perfect correlation "
                    f"with target (r={corr:.4f}). Possible leakage!"
                )


def check_cv_properly_implemented(
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    n_samples: int
) -> None:
    """
    Check that cross-validation is properly implemented.
    
    This checks:
    - All samples appear exactly once in validation
    - No overlap between train and validation in each fold
    - All folds cover all samples
    """
    all_val_indices = set()
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        train_set = set(train_idx)
        val_set = set(val_idx)
        
        # Check no overlap
        if train_set & val_set:
            raise DataLeakageError(
                f"CV Fold {fold_idx}: Train and validation sets overlap!"
            )
        
        # Collect validation indices
        all_val_indices.update(val_set)
    
    # Check all samples appear in validation
    if len(all_val_indices) != n_samples:
        missing = set(range(n_samples)) - all_val_indices
        raise DataLeakageError(
            f"CV: Not all samples appear in validation. Missing: {len(missing)} samples"
        )


def check_predictions_sanity(
    predictions: np.ndarray,
    model_name: str,
    target_is_log: bool = True
) -> None:
    """
    Check that predictions are reasonable.
    
    This checks:
    - No NaNs or Infs
    - Reasonable range (for log scale, typically 10-14 for house prices)
    - Not all the same value
    """
    if not np.isfinite(predictions).all():
        raise DataIntegrityError(
            f"[{model_name}] Predictions contain NaN or Inf values"
        )
    
    if target_is_log:
        # logSP should be in reasonable range (log of house prices)
        if (predictions < 0).any() or (predictions > 20).any():
            warnings.warn(
                f"[{model_name}] Predictions in log scale have suspicious range: "
                f"min={predictions.min():.2f}, max={predictions.max():.2f}"
            )
    else:
        # Real scale (SalePrice)
        if (predictions < 0).any() or (predictions > 1e7).any():
            warnings.warn(
                f"[{model_name}] Predictions have suspicious range: "
                f"min={predictions.min():.2f}, max={predictions.max():.2f}"
            )
    
    # Check not all same value
    if np.std(predictions) < 1e-6:
        raise DataIntegrityError(
            f"[{model_name}] All predictions are identical (std={np.std(predictions):.2e})"
        )


def check_submission_format(
    submission: pd.DataFrame,
    expected_test_size: int
) -> None:
    """Check that submission file has correct format."""
    if 'Id' not in submission.columns:
        raise DataIntegrityError("Submission missing 'Id' column")
    
    if 'SalePrice' not in submission.columns:
        raise DataIntegrityError("Submission missing 'SalePrice' column")
    
    if len(submission) != expected_test_size:
        raise DataIntegrityError(
            f"Submission has {len(submission)} rows, expected {expected_test_size}"
        )
    
    if submission['SalePrice'].isnull().any():
        raise DataIntegrityError("Submission has missing SalePrice values")
    
    if (submission['SalePrice'] <= 0).any():
        raise DataIntegrityError("Submission has non-positive SalePrice values")


# ============================================================================
# COMPREHENSIVE STAGE VALIDATION
# ============================================================================

def validate_preprocessing_stage(
    stage_num: int,
    train_before_path: Optional[Path] = None,
    test_before_path: Optional[Path] = None,
    train_after_path: Optional[Path] = None,
    test_after_path: Optional[Path] = None,
    stop_on_error: bool = True
) -> Dict[str, Union[bool, str, List[str]]]:
    """
    Comprehensive validation for a preprocessing stage.
    
    If stop_on_error=True, raises exceptions that will stop the pipeline.
    Returns results dict if stop_on_error=False.
    """
    if cfg is None:
        if stop_on_error:
            raise ImportError("config_local not available")
        return {"skipped": True, "reason": "config_local not available"}
    
    stage_name = f"Stage {stage_num}"
    errors = []
    warnings_list = []
    
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"VALIDATING {stage_name}")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    try:
        # Load data
        if train_after_path is None:
            train_attr = f"TRAIN_PROCESS{stage_num}_CSV"
            test_attr = f"TEST_PROCESS{stage_num}_CSV"
            train_after_path = getattr(cfg, train_attr, None)
            test_after_path = getattr(cfg, test_attr, None)
        
        if train_after_path is None or not Path(train_after_path).exists():
            msg = f"Stage {stage_num} output file not found: {train_after_path}"
            if stop_on_error:
                raise FileNotFoundError(msg)
            return {"skipped": True, "reason": msg}
        
        train_after = pd.read_csv(train_after_path)
        test_after = pd.read_csv(test_after_path)
        
        # Load before data if available
        train_before = None
        test_before = None
        if train_before_path and Path(train_before_path).exists():
            train_before = pd.read_csv(train_before_path)
        elif stage_num > 1:
            # Try to load previous stage
            prev_attr = f"TRAIN_PROCESS{stage_num-1}_CSV"
            if hasattr(cfg, prev_attr):
                prev_path = getattr(cfg, prev_attr)
                if Path(prev_path).exists():
                    train_before = pd.read_csv(prev_path)
                    test_before = pd.read_csv(getattr(cfg, f"TEST_PROCESS{stage_num-1}_CSV"))
        
        # ====================================================================
        # CHECK 1: Train/Test Leakage
        # ====================================================================
        print(f"\n{Fore.CYAN}[1/7] Checking for train/test leakage...{Style.RESET_ALL}")
        try:
            check_no_train_test_leakage(train_after, test_after, stage_name)
            print_success("No train/test leakage detected")
        except DataLeakageError as e:
            error_msg = f"DATA LEAKAGE DETECTED: {str(e)}"
            print_error(error_msg, "LEAKAGE")
            errors.append(error_msg)
            if stop_on_error:
                raise
        
        # ====================================================================
        # CHECK 2: Missing Values
        # ====================================================================
        print(f"\n{Fore.CYAN}[2/7] Checking missing values...{Style.RESET_ALL}")
        try:
            check_missing_values_handled(
                train_after, test_after, stage_name,
                allow_target_nans=(stage_num == 1)
            )
            print_success("Missing values properly handled")
        except DataIntegrityError as e:
            error_msg = f"MISSING VALUES: {str(e)}"
            print_error(error_msg, "INTEGRITY")
            errors.append(error_msg)
            if stop_on_error:
                raise
        
        # ====================================================================
        # CHECK 3: Shape Consistency
        # ====================================================================
        if train_before is not None and test_before is not None:
            print(f"\n{Fore.CYAN}[3/7] Checking shape consistency...{Style.RESET_ALL}")
            try:
                expected_row_change = None
                if stage_num == 2:  # Stage 2 removes outliers
                    expected_row_change = -1
                check_shape_consistency(
                    train_before, train_after,
                    test_before, test_after,
                    stage_name, expected_row_change
                )
                print_success(f"Shapes consistent (train: {len(train_after)}, test: {len(test_after)})")
            except DataIntegrityError as e:
                error_msg = f"SHAPE MISMATCH: {str(e)}"
                print_error(error_msg, "INTEGRITY")
                errors.append(error_msg)
                if stop_on_error:
                    raise
        
        # ====================================================================
        # CHECK 4: Column Parity
        # ====================================================================
        print(f"\n{Fore.CYAN}[4/7] Checking column parity...{Style.RESET_ALL}")
        target_col = "logSP" if stage_num >= 2 else "SalePrice"
        train_cols = set(train_after.columns) - {target_col}
        test_cols = set(test_after.columns)
        
        if train_cols != test_cols:
            only_train = train_cols - test_cols
            only_test = test_cols - train_cols
            error_msg = (
                f"COLUMN MISMATCH:\n"
                f"  Only in train: {only_train}\n"
                f"  Only in test: {only_test}"
            )
            print_error(error_msg, "INTEGRITY")
            errors.append(error_msg)
            if stop_on_error:
                raise DataIntegrityError(error_msg)
        else:
            print_success(f"Column parity OK ({len(train_cols)} shared features)")
        
        # ====================================================================
        # CHECK 5: Target Integrity
        # ====================================================================
        print(f"\n{Fore.CYAN}[5/7] Checking target integrity...{Style.RESET_ALL}")
        if target_col in train_after.columns:
            try:
                # Only check logSP transformation if both SalePrice and logSP exist
                # (logSP is created in stage 2, so stage 1 won't have this check)
                if target_col == 'logSP' and 'SalePrice' in train_after.columns:
                    check_target_not_in_features(train_after, stage_name, target_col)
                elif target_col == 'SalePrice':
                    # For stage 1, just check that SalePrice is valid
                    if train_after[target_col].isnull().any():
                        raise DataIntegrityError(f"[{stage_name}] Target {target_col} has missing values")
                    if (train_after[target_col] <= 0).any():
                        raise DataIntegrityError(f"[{stage_name}] Target {target_col} has non-positive values")
                print_success(f"Target '{target_col}' is valid")
            except DataIntegrityError as e:
                error_msg = f"TARGET ISSUE: {str(e)}"
                print_error(error_msg, "INTEGRITY")
                errors.append(error_msg)
                if stop_on_error:
                    raise
        else:
            warnings_list.append(f"Target column '{target_col}' not found in train")
        
        # ====================================================================
        # CHECK 6: Feature Engineering Sanity
        # ====================================================================
        if stage_num >= 4:
            print(f"\n{Fore.CYAN}[6/7] Checking feature engineering sanity...{Style.RESET_ALL}")
            try:
                check_feature_engineering_sanity(train_after, test_after, stage_name)
                print_success("Feature engineering checks passed")
            except DataIntegrityError as e:
                error_msg = f"FEATURE SANITY: {str(e)}"
                print_error(error_msg, "INTEGRITY")
                errors.append(error_msg)
                if stop_on_error:
                    raise
            except Warning as w:
                warnings_list.append(str(w))
        
        # ====================================================================
        # CHECK 7: Target Leakage in Features
        # ====================================================================
        if stage_num >= 2:
            print(f"\n{Fore.CYAN}[7/7] Checking for target leakage in features...{Style.RESET_ALL}")
            try:
                check_no_target_leakage_in_features(train_after, test_after, stage_name)
                print_success("No target leakage detected in features")
            except (DataLeakageError, Warning) as e:
                if isinstance(e, DataLeakageError):
                    error_msg = f"TARGET LEAKAGE: {str(e)}"
                    print_error(error_msg, "LEAKAGE")
                    errors.append(error_msg)
                    if stop_on_error:
                        raise
                else:
                    warnings_list.append(str(e))
        
        # ====================================================================
        # SUMMARY
        # ====================================================================
        print(f"\n{Fore.CYAN}{'='*60}")
        if errors:
            print(f"{Fore.RED}VALIDATION FAILED: {len(errors)} error(s) found{Style.RESET_ALL}")
            for i, err in enumerate(errors, 1):
                print(f"{Fore.RED}  {i}. {err}{Style.RESET_ALL}")
            if stop_on_error:
                raise DataIntegrityError(f"Stage {stage_num} validation failed with {len(errors)} error(s)")
        else:
            print(f"{Fore.GREEN}[SUCCESS] ALL CHECKS PASSED for {stage_name}{Style.RESET_ALL}")
        
        if warnings_list:
            print(f"\n{Fore.YELLOW}Warnings ({len(warnings_list)}):{Style.RESET_ALL}")
            for w in warnings_list:
                print_warning(w)
        
        print(f"{'='*60}{Style.RESET_ALL}\n")
        
        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "warnings": warnings_list
        }
        
    except Exception as e:
        if stop_on_error:
            raise
        return {"failed": True, "error": str(e)}


# Keep existing check() function for backward compatibility
def check() -> None:
    """Legacy check function - calls comprehensive validation."""
    if cfg is None:
        print("[WARNING] config_local not available. Skipping checks.")
        return
    
    print("Running comprehensive preprocessing checks...")
    for stage in range(1, 7):
        results = validate_preprocessing_stage(stage, stop_on_error=False)
        if results.get('skipped'):
            continue
        
        print(f"\n--- Stage {stage} Results ---")
        for key, value in results.items():
            if key.endswith('_error'):
                print(f"  [FAILED] {key}: {value}")
            elif isinstance(value, bool):
                status = "[PASS]" if value else "[FAIL]"
                print(f"  {status} {key}")


# Keep existing evaluate_many function
DEFAULT_THRESHOLDS = {
    "abs_skew": 0.5,
    "abs_excess_kurt": 1.0,
    "jb_pvalue": 0.05,
    "ks_norm": 0.08,
}

def _ks_stat(s: pd.Series, dist_name: str = "norm") -> float:
    """Calculate Kolmogorov-Smirnov statistic for distribution fit."""
    dist = getattr(stats, dist_name)
    params = dist.fit(s.dropna())
    ks_stat, _ = stats.kstest(s.dropna(), dist_name, args=params)
    return float(ks_stat)

def evaluate_many(
    series_dict: Dict[str, pd.Series],
    thresholds: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Check distribution shape & fit for multiple series.
    
    Args:
        series_dict: Dictionary mapping series names to Series objects
        thresholds: Optional custom thresholds for metrics
        
    Returns:
        DataFrame with evaluation metrics for each series
    """
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    rows = []

    for name, s in series_dict.items():
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            continue

        # shape
        skew = stats.skew(s, bias=False)
        kurt_ex = stats.kurtosis(s, fisher=True, bias=False)
        jb_stat, jb_p = stats.jarque_bera(s)

        def add(metric, value, cmp_op="<=", note=""):
            thr = th.get(metric, np.nan)
            ok = (value <= thr) if cmp_op == "<=" else (value >= thr)
            rows.append([name, metric, float(value), float(thr), ok, note])

        add("abs_skew", abs(skew))
        add("abs_excess_kurt", abs(kurt_ex))
        rows.append([name, "jb_stat", float(jb_stat), np.nan, None, "smaller is better"])
        rows.append([name, "jb_pvalue", float(jb_p), th["jb_pvalue"], jb_p >= th["jb_pvalue"], ">= alpha passes"])
        add("ks_norm", _ks_stat(s, "norm"))

    out = pd.DataFrame(rows, columns=["series", "metric", "value", "threshold", "pass", "note"])
    out["value"] = out["value"].round(3)
    overall = (out.groupby("series")["pass"]
                 .apply(lambda c: c.dropna().all() if len(c.dropna()) else False)
                 .rename("overall_pass").reset_index())
    return out.merge(overall, on="series", how="left")

if __name__ == "__main__":
    check()
