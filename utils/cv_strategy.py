"""
Cross-validation strategy utilities.

Implements stratified CV based on target quantiles to better reflect
Kaggle test distribution and prevent overfitting.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from typing import Tuple, List, Optional


def create_stratified_cv_splits(
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    n_bins: int = 10
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified CV splits based on target quantiles.
    
    This ensures each fold has a similar distribution of target values,
    which better reflects the Kaggle test distribution and prevents
    overfitting to specific price ranges.
    
    Args:
        y: Target values (in log space)
        n_splits: Number of CV folds
        random_state: Random state for reproducibility
        n_bins: Number of quantile bins for stratification
        
    Returns:
        List of (train_indices, val_indices) tuples
    """
    # Create quantile-based bins for stratification
    # Use pd.qcut-like logic to create roughly equal-sized bins
    sorted_indices = np.argsort(y)
    n_samples = len(y)
    
    # Create bins with roughly equal counts
    bin_size = n_samples // n_bins
    bins = np.zeros(n_samples, dtype=int)
    
    for i in range(n_bins):
        start_idx = i * bin_size
        if i == n_bins - 1:
            end_idx = n_samples
        else:
            end_idx = (i + 1) * bin_size
        bins[sorted_indices[start_idx:end_idx]] = i
    
    # Use StratifiedKFold on the bins
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )
    
    splits = list(skf.split(np.arange(n_samples), bins))
    
    return splits


def get_cv_strategy(
    strategy: str = "stratified",
    y: Optional[np.ndarray] = None,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    n_bins: int = 10
):
    """
    Get a CV splitter based on the specified strategy.
    
    Args:
        strategy: "stratified" (target quantiles) or "kfold" (standard KFold)
        y: Target values (required for stratified)
        n_splits: Number of CV folds
        shuffle: Whether to shuffle (for KFold)
        random_state: Random state
        n_bins: Number of quantile bins (for stratified)
        
    Returns:
        CV splitter object or list of splits
    """
    if strategy == "stratified":
        if y is None:
            raise ValueError("y is required for stratified CV")
        return create_stratified_cv_splits(
            y=y,
            n_splits=n_splits,
            random_state=random_state,
            n_bins=n_bins
        )
    elif strategy == "kfold":
        kf = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
        return kf
    else:
        raise ValueError(f"Unknown CV strategy: {strategy}. Use 'stratified' or 'kfold'")

