"""Model utilities for training and evaluation."""
from pathlib import Path
from typing import Optional
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

try:
    import config_local.local_config as cfg
except ImportError:
    cfg = None


def save_model(model: object, path: Path, method: str = "joblib") -> None:
    """
    Save trained model to disk.
    
    Args:
        model: Trained model object
        path: Path to save model
        method: Serialization method ('joblib' or 'pickle')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if method == "joblib":
        joblib.dump(model, path)
    else:
        with open(path, "wb") as f:
            pickle.dump(model, f)


def load_model(path: Path, method: str = "joblib") -> object:
    """
    Load trained model from disk.
    
    Args:
        path: Path to model file
        method: Serialization method ('joblib' or 'pickle')
        
    Returns:
        Loaded model object
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    if method == "joblib":
        return joblib.load(path)
    else:
        with open(path, "rb") as f:
            return pickle.load(f)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    log_space: bool = True
) -> dict:
    """
    Evaluate model predictions.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        log_space: If True, values are in log space (use expm1)
        
    Returns:
        Dictionary with evaluation metrics
    """
    if log_space:
        y_true_real = np.expm1(y_true)
        y_pred_real = np.expm1(y_pred)
    else:
        y_true_real = y_true
        y_pred_real = y_pred
    
    mse = mean_squared_error(y_true_real, y_pred_real)
    rmse = np.sqrt(mse)
    
    return {
        "mse": mse,
        "rmse": rmse,
        "log_space": log_space
    }


def create_submission(
    predictions: np.ndarray,
    test_ids: pd.Series,
    filename: str,
    log_space: bool = True,
    model_name: Optional[str] = None,
    validate: bool = True
) -> pd.DataFrame:
    """
    Create Kaggle submission file.
    Enhanced with validation and better path handling.
    
    Args:
        predictions: Model predictions
        test_ids: Test set IDs
        filename: Output filename
        log_space: If True, predictions are in log space
        model_name: Optional model name for better path organization
        validate: If True, validate submission format
        
    Returns:
        Submission DataFrame
        
    Raises:
        ImportError: If config_local is not available
        ValueError: If validation fails
    """
    if cfg is None:
        raise ImportError("config_local not available")
    
    # Convert predictions from log space if needed
    if log_space:
        predictions = np.expm1(predictions)
    
    # Validate predictions
    if validate:
        if len(predictions) != len(test_ids):
            raise ValueError(
                f"Predictions length ({len(predictions)}) doesn't match test_ids length ({len(test_ids)})"
            )
        if np.any(predictions <= 0):
            raise ValueError(f"Found {np.sum(predictions <= 0)} non-positive predictions")
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            raise ValueError("Found NaN or Inf values in predictions")
    
    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": predictions
    })
    
    # Determine output path
    if hasattr(cfg, "get_model_submission_path"):
        if model_name:
            output_path = cfg.get_model_submission_path(model_name, filename)
        else:
            # Attempt to infer model name from filename
            model_name = filename.split("_")[0].split(".")[0]
            output_path = cfg.get_model_submission_path(model_name, filename)
    else:
        output_path = cfg.SUBMISSIONS_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save submission
    submission.to_csv(output_path, index=False)
    
    if validate:
        print(f"[SUCCESS] Submission saved: {output_path}")
        print(f"  Predictions: {len(predictions)} rows")
        print(f"  Price range: ${predictions.min():,.0f} - ${predictions.max():,.0f}")
        print(f"  Mean price: ${predictions.mean():,.0f}")
    
    return submission

