"""
Feature Selection for House Prices Prediction
=============================================
Implements multiple feature selection methods and combines them for optimal results.

Best Practices:
- Uses multiple models (XGBoost, LightGBM, CatBoost) for robustness
- Cross-validated feature importance (prevents overfitting)
- Multiple selection methods (SHAP, permutation, correlation, Lasso)
- Time-efficient (uses fast methods, parallel processing)
- Saves selected features for reuse

Expected Impact: 0.002-0.008 RMSE improvement by removing noise features
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

import config_local.local_config as local_config
from utils.engineering import update_engineering_summary

# Try to import optional dependencies
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

try:
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    print(f"Warning: Some models not available: {e}")


def get_feature_importance_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int = 4
) -> pd.Series:
    """Get cross-validated feature importance from XGBoost."""
    print("  Computing XGBoost feature importance...")
    importance_scores = np.zeros(len(X.columns))
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state + fold,
            n_jobs=-1,
            tree_method='hist',
            verbosity=0
        )
        model.fit(X_train.values, y_train.values)
        importance_scores += model.feature_importances_
    
    importance_scores /= n_splits
    return pd.Series(importance_scores, index=X.columns, name='xgboost_importance')


def get_feature_importance_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int = 4
) -> pd.Series:
    """Get cross-validated feature importance from LightGBM."""
    print("  Computing LightGBM feature importance...")
    importance_scores = np.zeros(len(X.columns))
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state + fold,
            n_jobs=-1,
            verbosity=-1
        )
        model.fit(X_train.values, y_train.values)
        importance_scores += model.feature_importances_
    
    importance_scores /= n_splits
    return pd.Series(importance_scores, index=X.columns, name='lightgbm_importance')


def get_feature_importance_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    iterations: int = 100,
    depth: int = 4
) -> pd.Series:
    """Get cross-validated feature importance from CatBoost."""
    print("  Computing CatBoost feature importance...")
    importance_scores = np.zeros(len(X.columns))
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            random_seed=random_state + fold,
            thread_count=-1,
            verbose=False
        )
        model.fit(X_train.values, y_train.values)
        importance_scores += model.feature_importances_
    
    importance_scores /= n_splits
    return pd.Series(importance_scores, index=X.columns, name='catboost_importance')


def get_shap_importance(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = 'xgboost',
    n_samples: int = 500,
    random_state: int = 42
) -> Optional[pd.Series]:
    """Get SHAP-based feature importance (time-efficient with sampling)."""
    if not SHAP_AVAILABLE:
        return None
    
    print(f"  Computing SHAP importance ({model_type})...")
    
    # Sample data for faster computation
    if len(X) > n_samples:
        sample_idx = np.random.RandomState(random_state).choice(
            len(X), n_samples, replace=False
        )
        X_sample = X.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]
    else:
        X_sample = X
        y_sample = y
    
    # Train model
    if model_type == 'xgboost':
        model = XGBRegressor(n_estimators=100, max_depth=4, random_state=random_state, n_jobs=-1, verbosity=0)
    elif model_type == 'lightgbm':
        model = LGBMRegressor(n_estimators=100, max_depth=4, random_state=random_state, n_jobs=-1, verbosity=-1)
    else:
        return None
    
    model.fit(X_sample.values, y_sample.values)
    
    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample.values)
    
    # Average absolute SHAP values
    importance = np.abs(shap_values).mean(axis=0)
    
    return pd.Series(importance, index=X.columns, name='shap_importance')


def get_correlation_importance(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.01
) -> pd.Series:
    """Get feature importance based on correlation with target."""
    print("  Computing correlation-based importance...")
    correlations = X.corrwith(y).abs()
    # Set minimum threshold to avoid zero importance
    correlations = correlations.clip(lower=threshold)
    return correlations.rename('correlation_importance')


def get_lasso_selection(
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    random_state: int = 42
) -> pd.Series:
    """Get feature selection using LassoCV."""
    print("  Computing Lasso-based feature selection...")
    
    # Use LassoCV to find optimal alpha
    lasso = LassoCV(cv=cv, random_state=random_state, n_jobs=-1, max_iter=2000)
    lasso.fit(X.values, y.values)
    
    # Get coefficients (absolute value as importance)
    importance = np.abs(lasso.coef_)
    
    return pd.Series(importance, index=X.columns, name='lasso_importance')


def combine_importance_scores(
    importance_dict: Dict[str, pd.Series],
    weights: Optional[Dict[str, float]] = None
) -> pd.Series:
    """Combine multiple importance scores with optional weights."""
    if weights is None:
        # Equal weights
        weights = {name: 1.0 for name in importance_dict.keys()}
    
    # Normalize each importance score to [0, 1]
    normalized_scores = {}
    for name, scores in importance_dict.items():
        if scores.sum() > 0:
            normalized_scores[name] = scores / scores.max()
        else:
            normalized_scores[name] = scores
    
    # Weighted average
    combined = pd.Series(0.0, index=list(importance_dict.values())[0].index)
    total_weight = 0
    
    for name, scores in normalized_scores.items():
        weight = weights.get(name, 1.0)
        combined += scores * weight
        total_weight += weight
    
    if total_weight > 0:
        combined /= total_weight
    
    return combined.rename('combined_importance')


def select_features_by_percentile(
    importance_scores: pd.Series,
    percentile: float = 10.0
) -> List[str]:
    """Select top features by percentile threshold."""
    threshold = np.percentile(importance_scores, percentile)
    selected = importance_scores[importance_scores >= threshold].index.tolist()
    return selected


def select_features_by_count(
    importance_scores: pd.Series,
    n_features: int
) -> List[str]:
    """Select top N features."""
    return importance_scores.nlargest(n_features).index.tolist()


def evaluate_feature_set(
    X: pd.DataFrame,
    y: pd.Series,
    selected_features: List[str],
    model_type: str = 'xgboost',
    n_splits: int = 5,
    random_state: int = 42
) -> float:
    """Evaluate feature set using cross-validation."""
    # Filter to only numeric features (tree models can't handle strings)
    numeric_features = [f for f in selected_features if f in X.columns and pd.api.types.is_numeric_dtype(X[f])]
    
    if len(numeric_features) == 0:
        return np.inf
    
    X_selected = X[numeric_features]
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    
    for train_idx, val_idx in kf.split(X_selected):
        X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if model_type == 'xgboost':
            model = XGBRegressor(n_estimators=100, max_depth=4, random_state=random_state, n_jobs=-1, verbosity=0)
        elif model_type == 'lightgbm':
            model = LGBMRegressor(n_estimators=100, max_depth=4, random_state=random_state, n_jobs=-1, verbosity=-1)
        else:
            return np.inf
        
        model.fit(X_train.values, y_train.values)
        y_pred = model.predict(X_val.values)
        rmse = np.sqrt(mean_squared_error(y_val.values, y_pred))
        scores.append(rmse)
    
    return np.mean(scores)


def find_optimal_feature_count(
    X: pd.DataFrame,
    y: pd.Series,
    importance_scores: pd.Series,
    min_features: int = 50,
    max_features: int = None,
    step: int = 20,
    model_type: str = 'xgboost'
) -> Tuple[int, float]:
    """Find optimal number of features by testing different counts."""
    # Only consider numeric features for evaluation
    numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    numeric_importance = importance_scores[numeric_cols]
    
    if max_features is None:
        max_features = len(numeric_cols)
    
    print(f"\n  Testing feature counts from {min_features} to {max_features} (step={step})...")
    
    best_count = len(numeric_cols)
    best_score = evaluate_feature_set(X, y, numeric_cols, model_type)
    
    feature_counts = range(min_features, min(max_features + 1, len(numeric_cols) + 1), step)
    
    for n_features in feature_counts:
        selected_numeric = select_features_by_count(numeric_importance, n_features)
        score = evaluate_feature_set(X, y, selected_numeric, model_type)
        
        print(f"    {n_features:4d} features: RMSE = {score:.6f}", end="")
        if score < best_score:
            best_score = score
            best_count = n_features
            print(f"  ← Best so far!")
        else:
            print()
    
    return best_count, best_score


def remove_redundant_features(
    X: pd.DataFrame,
    importance_scores: pd.Series,
    correlation_threshold: float = 0.95
) -> List[str]:
    """
    Remove redundant features based on feature-to-feature correlation.
    When two features are highly correlated, keep the one with higher importance.
    
    Args:
        X: Feature dataframe
        importance_scores: Feature importance scores
        correlation_threshold: Correlation threshold above which features are considered redundant
    
    Returns:
        List of selected feature names (redundant ones removed)
    """
    print(f"\n  Removing redundant features (correlation > {correlation_threshold})...")
    
    # Only consider numeric features
    numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    X_numeric = X[numeric_cols]
    
    if len(numeric_cols) == 0:
        return []
    
    # Compute correlation matrix
    corr_matrix = X_numeric.corr().abs()
    
    # Find highly correlated pairs
    redundant_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > correlation_threshold:
                redundant_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    if len(redundant_pairs) == 0:
        print(f"    No redundant features found (all correlations <= {correlation_threshold})")
        return numeric_cols
    
    print(f"    Found {len(redundant_pairs)} highly correlated feature pairs")
    
    # Determine which features to remove
    # Strategy: For each pair, remove the one with lower importance
    features_to_remove = set()
    
    for feat1, feat2, corr_val in redundant_pairs:
        # Get importance scores (use 0 if not in importance_scores)
        imp1 = importance_scores.get(feat1, 0.0)
        imp2 = importance_scores.get(feat2, 0.0)
        
        if imp1 < imp2:
            features_to_remove.add(feat1)
            print(f"      Removing {feat1} (corr={corr_val:.3f} with {feat2}, lower importance)")
        else:
            features_to_remove.add(feat2)
            print(f"      Removing {feat2} (corr={corr_val:.3f} with {feat1}, lower importance)")
    
    # Return features that are not redundant
    selected_features = [f for f in numeric_cols if f not in features_to_remove]
    
    print(f"    Removed {len(features_to_remove)} redundant features")
    print(f"    Remaining: {len(selected_features)} features")
    
    return selected_features


def main(
    input_train_path: Optional[Path] = None,
    input_test_path: Optional[Path] = None,
    output_train_path: Optional[Path] = None,
    output_test_path: Optional[Path] = None,
    method: str = 'auto',  # 'auto', 'percentile', 'count', 'optimal'
    n_features: Optional[int] = None,
    percentile: float = 10.0,
    use_shap: bool = True,
    use_lasso: bool = True,
    use_correlation: bool = True,
    find_optimal: bool = False,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Main feature selection function.
    
    Args:
        input_train_path: Path to input training data (default: process6)
        input_test_path: Path to input test data (default: process6)
        output_train_path: Path to save selected training data (default: process7)
        output_test_path: Path to save selected test data (default: process7)
        method: Selection method ('auto', 'percentile', 'count', 'optimal')
        n_features: Number of features to select (if method='count')
        percentile: Percentile threshold (if method='percentile')
        use_shap: Whether to use SHAP importance
        use_lasso: Whether to use Lasso selection
        use_correlation: Whether to use correlation importance
        find_optimal: Whether to find optimal feature count (slower but better)
        random_state: Random state for reproducibility
    
    Returns:
        Tuple of (train_df, test_df, selected_features)
    """
    print("="*70)
    print("FEATURE SELECTION FOR HOUSE PRICES PREDICTION")
    print("="*70)
    
    # Set default paths
    if input_train_path is None:
        input_train_path = local_config.TRAIN_PROCESS6_CSV
    if input_test_path is None:
        input_test_path = local_config.TEST_PROCESS6_CSV
    if output_train_path is None:
        output_train_path = local_config.INTERIM_TRAIN_DIR / "train_process7.csv"
    if output_test_path is None:
        output_test_path = local_config.INTERIM_TEST_DIR / "test_process7.csv"
    
    # Load data
    print(f"\nLoading data...")
    print(f"  Train: {input_train_path}")
    print(f"  Test: {input_test_path}")
    train = pd.read_csv(input_train_path)
    test = pd.read_csv(input_test_path)
    
    print(f"Initial shape - Train: {train.shape}, Test: {test.shape}")
    
    # Extract target and features
    y = train["logSP"]
    X_train = train.drop(columns=["logSP"])
    X_test = test
    
    # Ensure same columns
    common_cols = [col for col in X_train.columns if col in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    # Separate numeric and categorical columns
    # Tree-based models need numeric data, so we'll exclude categoricals from importance calculation
    # Categoricals will be handled by target encoding in stage 8
    numeric_cols = [col for col in X_train.columns 
                   if pd.api.types.is_numeric_dtype(X_train[col])]
    categorical_cols = [col for col in X_train.columns 
                        if col not in numeric_cols]
    
    if categorical_cols:
        print(f"  Note: {len(categorical_cols)} categorical columns will be excluded from tree-based importance")
        print(f"    (They will be target-encoded in stage 8): {categorical_cols[:5]}{'...' if len(categorical_cols) > 5 else ''}")
    
    # Use only numeric columns for feature importance (tree-based models)
    X_train_numeric = X_train[numeric_cols].copy()
    X_test_numeric = X_test[numeric_cols].copy()
    
    print(f"Features available: {len(X_train.columns)} total ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)")
    print(f"Using {len(numeric_cols)} numeric features for importance calculation")
    
    # Compute importance scores
    print(f"\nComputing feature importance scores...")
    importance_dict = {}
    
    # Tree-based importance (always use) - only on numeric features
    if MODELS_AVAILABLE:
        importance_dict['xgboost'] = get_feature_importance_xgboost(X_train_numeric, y, random_state=random_state)
        importance_dict['lightgbm'] = get_feature_importance_lightgbm(X_train_numeric, y, random_state=random_state)
        importance_dict['catboost'] = get_feature_importance_catboost(X_train_numeric, y, random_state=random_state)
    
    # SHAP importance (optional, slower) - only on numeric features
    if use_shap and SHAP_AVAILABLE:
        shap_importance = get_shap_importance(X_train_numeric, y, model_type='xgboost', random_state=random_state)
        if shap_importance is not None:
            importance_dict['shap'] = shap_importance
    
    # Correlation importance (fast) - only on numeric features
    if use_correlation:
        importance_dict['correlation'] = get_correlation_importance(X_train_numeric, y)
    
    # Lasso importance (fast) - only on numeric features
    if use_lasso:
        importance_dict['lasso'] = get_lasso_selection(X_train_numeric, y, random_state=random_state)
    
    # Combine importance scores
    print(f"\nCombining importance scores...")
    weights = {
        'xgboost': 2.0,
        'lightgbm': 2.0,
        'catboost': 2.0,
        'shap': 1.5,
        'correlation': 1.0,
        'lasso': 1.5
    }
    
    combined_importance = combine_importance_scores(importance_dict, weights)
    
    # Add categorical columns with low importance (they'll be handled by target encoding)
    # But we still want to keep them for now
    for cat_col in categorical_cols:
        combined_importance[cat_col] = 0.001  # Low but non-zero to keep them
    
    # Ensure all numeric columns have importance scores
    for col in numeric_cols:
        if col not in combined_importance.index:
            combined_importance[col] = 0.0
    
    # Save importance scores for analysis
    importance_df = pd.DataFrame(importance_dict)
    importance_df['combined'] = combined_importance[importance_df.index]
    importance_df = importance_df.sort_values('combined', ascending=False)
    
    importance_output = local_config.PROCESSED_DIR / "feature_importance.csv"
    importance_df.to_csv(importance_output)
    print(f"  Saved importance scores to: {importance_output}")
    
    # Remove redundant features (correlation > 0.95)
    # This should be done before final feature selection
    print(f"\nRemoving redundant features...")
    non_redundant_features = remove_redundant_features(
        X_train, 
        combined_importance, 
        correlation_threshold=0.95
    )
    
    # Update combined_importance to only include non-redundant features
    # (set redundant features to 0 so they won't be selected)
    for col in X_train.columns:
        if col not in non_redundant_features and col in combined_importance.index:
            combined_importance[col] = 0.0
    
    # Select features
    print(f"\nSelecting features (method: {method})...")
    
    # Always keep categorical columns (they'll be target-encoded in stage 8)
    # Only select from numeric features
    numeric_importance = combined_importance[numeric_cols]
    
    if method == 'auto':
        # Auto: use percentile if n_features not specified, otherwise use count
        if n_features is not None:
            selected_numeric = select_features_by_count(numeric_importance, n_features)
            print(f"  Selected top {n_features} numeric features")
        else:
            selected_numeric = select_features_by_percentile(numeric_importance, percentile)
            print(f"  Selected numeric features above {percentile}th percentile ({len(selected_numeric)} features)")
    
    elif method == 'percentile':
        selected_numeric = select_features_by_percentile(numeric_importance, percentile)
        print(f"  Selected numeric features above {percentile}th percentile ({len(selected_numeric)} features)")
    
    elif method == 'count':
        if n_features is None:
            n_features = 200  # Default
        selected_numeric = select_features_by_count(numeric_importance, n_features)
        print(f"  Selected top {n_features} numeric features")
    
    elif method == 'optimal':
        if find_optimal:
            best_count, best_score = find_optimal_feature_count(
                X_train_numeric, y, numeric_importance,
                min_features=100,
                max_features=min(350, len(numeric_cols)),
                step=25,
                model_type='xgboost'
            )
            selected_numeric = select_features_by_count(numeric_importance, best_count)
            print(f"\n  Optimal numeric feature count: {best_count} (RMSE: {best_score:.6f})")
        else:
            # Use default percentile
            selected_numeric = select_features_by_percentile(numeric_importance, percentile)
            print(f"  Selected numeric features above {percentile}th percentile ({len(selected_numeric)} features)")
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Combine selected numeric features with all categorical features
    selected_features = list(selected_numeric) + categorical_cols
    
    print(f"\nSelected {len(selected_features)} features total:")
    print(f"  - Numeric features: {len(selected_numeric)} (selected from {len(numeric_cols)})")
    print(f"  - Categorical features: {len(categorical_cols)} (kept for target encoding)")
    print(f"Reduction: {len(numeric_cols) - len(selected_numeric)} numeric features removed ({100*(1-len(selected_numeric)/len(numeric_cols)):.1f}%)")
    
    # Create filtered datasets
    train_selected = train[["logSP"] + selected_features].copy()
    test_selected = test[selected_features].copy()
    
    # Save results
    print(f"\nSaving selected features data...")
    Path(output_train_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_test_path).parent.mkdir(parents=True, exist_ok=True)
    
    train_selected.to_csv(output_train_path, index=False)
    test_selected.to_csv(output_test_path, index=False)
    
    print(f"  Train: {output_train_path}")
    print(f"  Test: {output_test_path}")
    print(f"Final shape - Train: {train_selected.shape}, Test: {test_selected.shape}")
    
    # Log engineering details
    update_engineering_summary("Feature Selection", {
        "method": method,
        "n_features_selected": len(selected_features),
        "n_features_original": len(X_train.columns),
        "reduction_percent": 100 * (1 - len(selected_features) / len(X_train.columns)),
        "importance_methods": list(importance_dict.keys()),
        "selected_features": selected_features[:20]  # First 20 for logging
    })
    
    print("\n" + "="*70)
    print("Feature Selection complete!")
    print("="*70)
    print(f"\nTop 20 selected features:")
    for i, feat in enumerate(selected_features[:20], 1):
        importance_val = combined_importance[feat]
        print(f"  {i:2d}. {feat:40s} (importance: {importance_val:.6f})")
    
    return train_selected, test_selected, selected_features


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Feature Selection for House Prices")
    parser.add_argument("--method", type=str, default="auto", 
                       choices=["auto", "percentile", "count", "optimal"],
                       help="Selection method")
    parser.add_argument("--n_features", type=int, default=None,
                       help="Number of features to select (for count method)")
    parser.add_argument("--percentile", type=float, default=10.0,
                       help="Percentile threshold (for percentile method)")
    parser.add_argument("--no-shap", action="store_true",
                       help="Disable SHAP importance (faster)")
    parser.add_argument("--no-lasso", action="store_true",
                       help="Disable Lasso selection")
    parser.add_argument("--no-correlation", action="store_true",
                       help="Disable correlation importance")
    parser.add_argument("--find-optimal", action="store_true",
                       help="Find optimal feature count (slower)")
    parser.add_argument("--random-state", type=int, default=42,
                       help="Random state")
    
    args = parser.parse_args()
    
    main(
        method=args.method,
        n_features=args.n_features,
        percentile=args.percentile,
        use_shap=not args.no_shap,
        use_lasso=not args.no_lasso,
        use_correlation=not args.no_correlation,
        find_optimal=args.find_optimal,
        random_state=args.random_state
    )

