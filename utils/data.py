"""Data loading and saving utilities."""
from pathlib import Path
from typing import Optional
import pandas as pd

try:
    import config_local.local_config as cfg
except ImportError:
    cfg = None


def load_data(stage: int, train: bool = True) -> pd.DataFrame:
    """
    Load processed data at specified stage.
    
    Args:
        stage: Processing stage (1-6)
        train: If True, load training data; otherwise test data
        
    Returns:
        DataFrame with processed data
    """
    if cfg is None:
        raise ImportError("config_local not available")
    
    prefix = "TRAIN" if train else "TEST"
    attr_name = f"{prefix}_PROCESS{stage}_CSV"
    
    if not hasattr(cfg, attr_name):
        raise ValueError(f"Stage {stage} not found in config")
    
    path = getattr(cfg, attr_name)
    if not Path(path).exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    return pd.read_csv(path)


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


def validate_data(df: pd.DataFrame, expected_columns: Optional[list] = None) -> bool:
    """
    Validate DataFrame structure.
    
    Args:
        df: DataFrame to validate
        expected_columns: Optional list of expected column names
        
    Returns:
        True if validation passes
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if expected_columns:
        missing = set(expected_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
    
    return True

