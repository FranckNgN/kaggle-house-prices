"""Model validation utilities for sanity checks during model training."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from utils.checks import (
    check_model_no_target_leakage,
    check_predictions_sanity,
    check_submission_format,
    check_cv_properly_implemented,
    DataLeakageError,
    DataIntegrityError,
    print_error,
    print_success,
    print_warning
)

try:
    from colorama import Fore, Style
except ImportError:
    class Fore:
        RED = ''
        GREEN = ''
        YELLOW = ''
        CYAN = ''
    class Style:
        RESET_ALL = ''


def validate_model_inputs(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    model_name: str
) -> bool:
    """
    Validate model inputs before training.
    
    Returns:
        True if all checks pass, raises exception otherwise
    """
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"VALIDATING INPUTS FOR {model_name.upper()}")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    errors = []
    
    # Check 1: No missing values
    print(f"\n{Fore.CYAN}[1/4] Checking for missing values...{Style.RESET_ALL}")
    train_nulls = X_train.isnull().sum().sum()
    test_nulls = X_test.isnull().sum().sum()
    y_nulls = y_train.isnull().sum()
    
    if train_nulls > 0:
        errors.append(f"X_train has {train_nulls} missing values")
    if test_nulls > 0:
        errors.append(f"X_test has {test_nulls} missing values")
    if y_nulls > 0:
        errors.append(f"y_train has {y_nulls} missing values")
    
    if errors:
        for err in errors:
            print_error(err, "INTEGRITY")
        raise DataIntegrityError(f"Missing values detected: {errors}")
    print_success("No missing values found")
    
    # Check 2: Target not in features
    print(f"\n{Fore.CYAN}[2/4] Checking for target leakage...{Style.RESET_ALL}")
    try:
        check_model_no_target_leakage(X_train, y_train, X_test, model_name)
        print_success("No target leakage detected")
    except DataLeakageError as e:
        print_error(str(e), "LEAKAGE")
        raise
    
    # Check 3: Column parity
    print(f"\n{Fore.CYAN}[3/4] Checking column parity...{Style.RESET_ALL}")
    train_cols = set(X_train.columns)
    test_cols = set(X_test.columns)
    
    if train_cols != test_cols:
        only_train = train_cols - test_cols
        only_test = test_cols - train_cols
        error_msg = (
            f"Column mismatch:\n"
            f"  Only in train: {only_train}\n"
            f"  Only in test: {only_test}"
        )
        print_error(error_msg, "INTEGRITY")
        raise DataIntegrityError(error_msg)
    print_success(f"Column parity OK ({len(train_cols)} features)")
    
    # Check 4: Data shapes
    print(f"\n{Fore.CYAN}[4/4] Checking data shapes...{Style.RESET_ALL}")
    if len(X_train) != len(y_train):
        error_msg = f"X_train and y_train have different lengths: {len(X_train)} vs {len(y_train)}"
        print_error(error_msg, "INTEGRITY")
        raise DataIntegrityError(error_msg)
    
    print_success(f"Shapes OK - Train: {X_train.shape}, Test: {X_test.shape}, Target: {len(y_train)}")
    
    print(f"\n{Fore.GREEN}✅ ALL INPUT VALIDATION CHECKS PASSED{Style.RESET_ALL}\n")
    return True


def validate_model_predictions(
    predictions: np.ndarray,
    model_name: str,
    target_is_log: bool = True
) -> bool:
    """
    Validate model predictions.
    
    Returns:
        True if all checks pass, raises exception otherwise
    """
    print(f"\n{Fore.CYAN}Validating predictions for {model_name}...{Style.RESET_ALL}")
    
    try:
        check_predictions_sanity(predictions, model_name, target_is_log=target_is_log)
        print_success("Predictions are valid")
        return True
    except DataIntegrityError as e:
        print_error(str(e), "INTEGRITY")
        raise


def validate_submission(
    submission: pd.DataFrame,
    expected_test_size: int,
    model_name: str,
    test_ids: Optional[pd.Series] = None
) -> bool:
    """
    Validate submission file format and ID matching.
    
    Args:
        submission: Submission dataframe
        expected_test_size: Expected number of test samples
        model_name: Name of the model
        test_ids: Optional Series of test IDs to match against
    
    Returns:
        True if all checks pass, raises exception otherwise
    """
    print(f"\n{Fore.CYAN}Validating submission for {model_name}...{Style.RESET_ALL}")
    
    try:
        check_submission_format(submission, expected_test_size)
        print_success("Submission format is valid")
        
        # Check ID matching if test_ids provided
        if test_ids is not None:
            print(f"\n{Fore.CYAN}Checking ID alignment...{Style.RESET_ALL}")
            if 'Id' not in submission.columns:
                raise DataIntegrityError("Submission missing 'Id' column for ID matching")
            
            submission_ids = submission['Id'].values
            test_ids_array = test_ids.values if hasattr(test_ids, 'values') else test_ids
            
            # Check same length
            if len(submission_ids) != len(test_ids_array):
                raise DataIntegrityError(
                    f"ID count mismatch: submission has {len(submission_ids)} IDs, "
                    f"test has {len(test_ids_array)} IDs"
                )
            
            # Check IDs match (order and values)
            if not np.array_equal(submission_ids, test_ids_array):
                # Check if it's just order issue
                if set(submission_ids) == set(test_ids_array):
                    raise DataIntegrityError(
                        f"ID order mismatch: submission IDs don't match test ID order"
                    )
                else:
                    missing = set(test_ids_array) - set(submission_ids)
                    extra = set(submission_ids) - set(test_ids_array)
                    error_msg = f"ID mismatch:\n"
                    if missing:
                        error_msg += f"  Missing IDs in submission: {list(missing)[:10]}{'...' if len(missing) > 10 else ''}\n"
                    if extra:
                        error_msg += f"  Extra IDs in submission: {list(extra)[:10]}{'...' if len(extra) > 10 else ''}"
                    raise DataIntegrityError(error_msg)
            
            print_success(f"ID alignment OK ({len(submission_ids)} IDs match)")
        
        return True
    except DataIntegrityError as e:
        print_error(str(e), "INTEGRITY")
        raise


def validate_cv_splits(
    cv_splits: list,
    n_samples: int,
    model_name: str
) -> bool:
    """
    Validate cross-validation splits.
    
    Returns:
        True if all checks pass, raises exception otherwise
    """
    print(f"\n{Fore.CYAN}Validating CV splits for {model_name}...{Style.RESET_ALL}")
    
    try:
        check_cv_properly_implemented(cv_splits, n_samples)
        print_success("CV splits are valid")
        return True
    except DataLeakageError as e:
        print_error(str(e), "LEAKAGE")
        raise

