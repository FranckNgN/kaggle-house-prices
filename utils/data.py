"""
Data loading and saving utilities.
Enhanced with better error handling and QoL improvements.
"""
from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np

try:
    import config_local.local_config as cfg
except ImportError:
    cfg = None


def load_data(stage: int, train: bool = True) -> pd.DataFrame:
    """
    Load processed data at specified stage.
    Enhanced with better error handling and validation.
    
    Args:
        stage: Processing stage (1-8)
        train: If True, load training data; otherwise test data
        
    Returns:
        DataFrame with processed data
        
    Raises:
        ImportError: If config_local is not available
        ValueError: If stage is invalid
        FileNotFoundError: If data file doesn't exist
    """
    if cfg is None:
        raise ImportError("config_local not available")
    
    if stage < 1 or stage > 8:
        raise ValueError(f"Stage must be between 1 and 8, got {stage}")
    
    prefix = "TRAIN" if train else "TEST"
    attr_name = f"{prefix}_PROCESS{stage}_CSV"
    
    if not hasattr(cfg, attr_name):
        raise ValueError(f"Stage {stage} not found in config. Available stages: 1-8")
    
    path = Path(getattr(cfg, attr_name))
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Please run preprocessing first: python notebooks/preprocessing/run_preprocessing.py"
        )
    
    try:
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"Data file is empty: {path}")
        return df
    except pd.errors.EmptyDataError:
        raise ValueError(f"Data file is empty: {path}")
    except Exception as e:
        raise IOError(f"Error reading data file {path}: {e}")


def load_sample_submission() -> pd.DataFrame:
    """
    Load the sample submission file to use as a template for new submissions.
    Enhanced with validation.
    
    Returns:
        DataFrame with 'Id' and 'SalePrice' columns from sample_submission.csv
        
    Raises:
        ImportError: If config_local is not available
        FileNotFoundError: If sample submission file doesn't exist
        ValueError: If file format is invalid
    """
    if cfg is None:
        raise ImportError("config_local not available")
    
    path = Path(cfg.SAMPLE_SUBMISSION_CSV)
    if not path.exists():
        raise FileNotFoundError(
            f"Sample submission file not found: {path}\n"
            f"Please download it from Kaggle competition page."
        )
    
    try:
        df = pd.read_csv(path)
        if 'Id' not in df.columns or 'SalePrice' not in df.columns:
            raise ValueError(
                f"Invalid submission format. Expected columns: 'Id', 'SalePrice'. "
                f"Found: {list(df.columns)}"
            )
        return df
    except Exception as e:
        raise IOError(f"Error reading sample submission file {path}: {e}")


def save_data(df: pd.DataFrame, stage: int, train: bool = True) -> None:
    """
    Save processed data at specified stage.
    
    Args:
        df: DataFrame to save
        stage: Processing stage (1-6)
        train: If True, save as training data; otherwise test data
    """
    if cfg is None:
        raise ImportError("config_local not available")
    
    prefix = "TRAIN" if train else "TEST"
    attr_name = f"{prefix}_PROCESS{stage}_CSV"
    
    if not hasattr(cfg, attr_name):
        raise ValueError(f"Stage {stage} not found in config")
    
    path = getattr(cfg, attr_name)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def validate_data(df: pd.DataFrame, expected_columns: Optional[List[str]] = None) -> bool:
    """
    Validate DataFrame structure.
    Enhanced with more comprehensive checks.
    
    Args:
        df: DataFrame to validate
        expected_columns: Optional list of expected column names
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    if df is None:
        raise ValueError("DataFrame is None")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if expected_columns:
        missing = set(expected_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        extra = set(df.columns) - set(expected_columns)
        if extra:
            # Warning only, not an error
            print(f"[WARNING] Extra columns found: {extra}")
    
    # Check for all-NaN columns
    nan_cols = df.columns[df.isnull().all()].tolist()
    if nan_cols:
        print(f"[WARNING] Columns with all NaN values: {nan_cols}")
    
    return True


def load_train_test_data(stage: int = 8) -> tuple:
    """
    Load both training and test data at specified stage.
    Convenience function for model training.
    
    Args:
        stage: Processing stage (default: 8, final stage)
        
    Returns:
        Tuple of (train_df, test_df)
    """
    train_df = load_data(stage, train=True)
    test_df = load_data(stage, train=False)
    return train_df, test_df

