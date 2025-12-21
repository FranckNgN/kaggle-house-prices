"""
Cross-validation strategy utilities.

Implements multiple CV strategies:
- Stratified CV based on target quantiles
- GroupKFold by neighborhood (prevents data leakage from neighborhood effects)
- Standard KFold
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
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


def create_groupkfold_splits(
    groups: np.ndarray,
    n_splits: int = 5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create GroupKFold splits based on group labels (e.g., Neighborhood).
    
    This ensures that all samples from the same group (neighborhood) are
    in the same fold, preventing data leakage from group-level effects.
    
    Args:
        groups: Group labels for each sample (e.g., neighborhood names/IDs)
        n_splits: Number of CV folds
        
    Returns:
        List of (train_indices, val_indices) tuples
    """
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(np.arange(len(groups)), groups=groups))
    return splits


def get_cv_strategy(
    strategy: str = "stratified",
    y: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    n_bins: int = 10
):
    """
    Get a CV splitter based on the specified strategy.
    
    Args:
        strategy: "stratified" (target quantiles), "group" (GroupKFold), 
                 "kfold" (standard KFold), or "repeated_stratified" (repeated stratified)
        y: Target values (required for stratified)
        groups: Group labels for GroupKFold (e.g., neighborhood IDs)
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
    elif strategy == "group":
        if groups is None:
            raise ValueError("groups is required for GroupKFold")
        return create_groupkfold_splits(
            groups=groups,
            n_splits=n_splits
        )
    elif strategy == "repeated_stratified":
        if y is None:
            raise ValueError("y is required for repeated stratified CV")
        # Create multiple stratified splits with different random states
        all_splits = []
        n_repeats = 3  # 3 repeats
        for repeat in range(n_repeats):
            splits = create_stratified_cv_splits(
                y=y,
                n_splits=n_splits,
                random_state=random_state + repeat,
                n_bins=n_bins
            )
            all_splits.extend(splits)
        return all_splits
    elif strategy == "kfold":
        kf = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
        return kf
    else:
        raise ValueError(
            f"Unknown CV strategy: {strategy}. "
            f"Use 'stratified', 'group', 'repeated_stratified', or 'kfold'"
        )

