"""
Model wrapper with automatic sanity checks.
Models can use this to automatically validate inputs, predictions, and submissions.
"""
import os
import pandas as pd
import numpy as np
from typing import Optional

# Check if validation is enabled (can be set via environment variable)
ENABLE_VALIDATION = os.getenv("ENABLE_MODEL_VALIDATION", "1").lower() in ("1", "true", "yes")

if ENABLE_VALIDATION:
    try:
        from utils.model_validation import (
            validate_model_inputs,
            validate_model_predictions,
            validate_submission,
            validate_cv_splits
        )
        VALIDATION_AVAILABLE = True
    except ImportError:
        VALIDATION_AVAILABLE = False
        print("⚠️  Model validation utilities not available. Running without validation.")
else:
    VALIDATION_AVAILABLE = False


def validate_inputs_wrapper(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    model_name: str,
    skip_if_disabled: bool = True
) -> bool:
    """
    Wrapper to validate model inputs.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        model_name: Name of the model
        skip_if_disabled: If True, silently skip if validation is disabled
        
    Returns:
        True if validation passed or was skipped
    """
    if not ENABLE_VALIDATION or not VALIDATION_AVAILABLE:
        if not skip_if_disabled:
            print(f"⚠️  Validation disabled for {model_name}")
        return True
    
    return validate_model_inputs(X_train, y_train, X_test, model_name)


def validate_predictions_wrapper(
    predictions: np.ndarray,
    model_name: str,
    target_is_log: bool = True,
    skip_if_disabled: bool = True
) -> bool:
    """
    Wrapper to validate model predictions.
    
    Args:
        predictions: Model predictions
        model_name: Name of the model
        target_is_log: Whether predictions are in log scale
        skip_if_disabled: If True, silently skip if validation is disabled
        
    Returns:
        True if validation passed or was skipped
    """
    if not ENABLE_VALIDATION or not VALIDATION_AVAILABLE:
        if not skip_if_disabled:
            print(f"⚠️  Validation disabled for {model_name}")
        return True
    
    return validate_model_predictions(predictions, model_name, target_is_log)


def validate_submission_wrapper(
    submission: pd.DataFrame,
    expected_test_size: int,
    model_name: str,
    test_ids: Optional[pd.Series] = None,
    skip_if_disabled: bool = True
) -> bool:
    """
    Wrapper to validate submission format and ID matching.
    
    Args:
        submission: Submission dataframe
        expected_test_size: Expected number of test samples
        model_name: Name of the model
        test_ids: Optional Series of test IDs to match against
        skip_if_disabled: If True, silently skip if validation is disabled
        
    Returns:
        True if validation passed or was skipped
    """
    if not ENABLE_VALIDATION or not VALIDATION_AVAILABLE:
        if not skip_if_disabled:
            print(f"⚠️  Validation disabled for {model_name}")
        return True
    
    return validate_submission(submission, expected_test_size, model_name, test_ids)


def validate_cv_wrapper(
    cv_splits: list,
    n_samples: int,
    model_name: str,
    skip_if_disabled: bool = True
) -> bool:
    """
    Wrapper to validate CV splits.
    
    Args:
        cv_splits: List of (train_idx, val_idx) tuples
        n_samples: Total number of samples
        model_name: Name of the model
        skip_if_disabled: If True, silently skip if validation is disabled
        
    Returns:
        True if validation passed or was skipped
    """
    if not ENABLE_VALIDATION or not VALIDATION_AVAILABLE:
        if not skip_if_disabled:
            print(f"⚠️  Validation disabled for {model_name}")
        return True
    
    return validate_cv_splits(cv_splits, n_samples, model_name)

