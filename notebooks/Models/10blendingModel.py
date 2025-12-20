"""Model blending script for ensemble predictions."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

from config_local import local_config, model_config
from utils.model_wrapper import validate_submission_wrapper


def load_oof_predictions(cfg: dict) -> Optional[Dict[str, np.ndarray]]:
    """Load out-of-fold predictions if available."""
    oof_predictions = {}
    model_name_mapping = {
        "xgb": "xgboost",
        "lgb": "lightgbm", 
        "cat": "catboost",
        "ridge": "ridge",
        "lasso": "lasso",
        "elasticNet": "elastic_net",
        "rf": "random_forest",
        "svr": "svr"
    }
    
    for blend_name, model_name in model_name_mapping.items():
        oof_path = local_config.OOF_DIR / f"{model_name}_oof_train.npy"
        if oof_path.exists():
            oof_predictions[blend_name] = np.load(oof_path)
            print(f"  - Loaded OOF predictions for {blend_name} ({model_name})")
    
    return oof_predictions if oof_predictions else None


def optimize_blending_weights(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    initial_weights: Optional[Dict[str, float]] = None,
    method: str = "SLSQP"
) -> Dict[str, float]:
    """
    Find optimal weights for blending using scipy.optimize.
    
    Args:
        predictions: Dictionary of model_name -> predictions array
        y_true: True target values (in log space)
        initial_weights: Optional initial weights to start optimization
        method: Optimization method ('SLSQP', 'L-BFGS-B', or 'trust-constr')
    
    Returns:
        Dictionary of optimized weights
    """
    model_names = list(predictions.keys())
    n_models = len(model_names)
    
    # Prepare prediction matrix
    pred_matrix = np.column_stack([predictions[name] for name in model_names])
    
    # Ensure predictions are in log space (if they're in real space, convert)
    # Check if predictions are in log space by comparing scale
    if pred_matrix.mean() > 1000:  # Likely in real space
        print("  Converting predictions from real space to log space...")
        pred_matrix = np.log1p(pred_matrix)
    
    def objective(weights):
        """Objective function: RMSE of blended predictions."""
        # Normalize weights to sum to 1
        weights = np.maximum(weights, 0)  # Ensure non-negative
        weights = weights / (weights.sum() + 1e-10)  # Normalize
        
        # Blend predictions
        blended = pred_matrix @ weights
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_true, blended))
        return rmse
    
    # Initial weights (equal weights or provided)
    if initial_weights:
        x0 = np.array([initial_weights.get(name, 1.0) for name in model_names])
    else:
        x0 = np.ones(n_models) / n_models  # Equal weights
    
    # Constraints: weights must sum to 1 and be non-negative
    constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1.0}
    bounds = [(0.0, None) for _ in range(n_models)]  # Non-negative weights
    
    print(f"\nOptimizing blending weights using {method}...")
    print(f"  Models: {', '.join(model_names)}")
    print(f"  Initial weights: {dict(zip(model_names, x0))}")
    
    # Optimize
    result = minimize(
        objective,
        x0,
        method=method,
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    if not result.success:
        print(f"  Warning: Optimization did not converge: {result.message}")
        print(f"  Using best found weights...")
    
    # Normalize final weights
    optimal_weights = np.maximum(result.x, 0)
    optimal_weights = optimal_weights / (optimal_weights.sum() + 1e-10)
    
    # Calculate final RMSE
    final_rmse = objective(optimal_weights)
    
    print(f"\n{'='*60}")
    print("OPTIMAL BLENDING WEIGHTS")
    print(f"{'='*60}")
    for name, weight in zip(model_names, optimal_weights):
        print(f"  {name:15s}: {weight:8.4f} ({weight*100:5.2f}%)")
    print(f"{'='*60}")
    print(f"Optimized RMSE: {final_rmse:.6f}")
    print(f"{'='*60}\n")
    
    return dict(zip(model_names, optimal_weights))


def load_predictions(cfg: dict) -> Dict[str, pd.DataFrame]:
    """Load all model predictions that have a non-zero weight."""
    predictions = {}
    models = cfg["models"]
    weights = cfg["weights"]
    
    # Only load models that have a weight > 0
    active_models = {name: filename for name, filename in models.items() 
                     if weights.get(name, 0) > 0}
    
    if not active_models:
        raise ValueError("No active models (weight > 0) found in configuration.")
        
    for name, filename in active_models.items():
        # Look for the file in the root SUBMISSIONS_DIR or its subfolders
        path = local_config.SUBMISSIONS_DIR / filename
        if not path.exists():
            # Try to find it in subfolders
            potential_paths = list(local_config.SUBMISSIONS_DIR.rglob(filename))
            if potential_paths:
                path = potential_paths[0]
            else:
                print(f"Warning: Prediction file not found: {filename} in {local_config.SUBMISSIONS_DIR} or its subfolders. Skipping model: {name}")
                continue
        
        predictions[name] = pd.read_csv(path)
    
    if not predictions:
        raise FileNotFoundError("No prediction files found in submissions directory.")
        
    return predictions


def validate_alignment(predictions: Dict[str, pd.DataFrame]) -> None:
    """Validate that all predictions have aligned Id columns."""
    model_names = list(predictions.keys())
    first_model = model_names[0]
    first_ids = predictions[first_model]["Id"]
    
    for name in model_names[1:]:
        if not first_ids.equals(predictions[name]["Id"]):
            raise ValueError(f"Id alignment mismatch between {name} and {first_model}")


def blend_predictions(
    predictions: Dict[str, pd.DataFrame],
    weights: Dict[str, float]
) -> pd.DataFrame:
    """
    Blend predictions using weighted average.
    """
    # Use only models we successfully loaded
    active_model_names = list(predictions.keys())
    
    # Filter weights to only include loaded models and normalize
    active_weights = {name: weights[name] for name in active_model_names}
    total_weight = sum(active_weights.values())
    
    if total_weight == 0:
        raise ValueError("Total weight of loaded models is zero.")
        
    normalized_weights = {k: v / total_weight for k, v in active_weights.items()}
    
    # Initialize blended predictions with the first available model
    first_name = active_model_names[0]
    blend = predictions[first_name].copy()
    
    # Start SalePrice at zero to build the weighted sum
    blend["SalePrice"] = 0.0
    
    # Weighted average
    for name, pred in predictions.items():
        weight = normalized_weights[name]
        blend["SalePrice"] += weight * pred["SalePrice"]
        print(f"  - Applied {name} with normalized weight: {weight:.4f}")
    
    return blend


def main(optimize_weights: bool = True) -> None:
    """
    Main entry point for blending.
    
    Args:
        optimize_weights: If True, automatically optimize weights using OOF/validation data
    """
    cfg = model_config.BLENDING
    
    # Load predictions
    print("--- Loading predictions ---")
    predictions = load_predictions(cfg)
    
    # Validate alignment
    print("--- Validating alignment ---")
    validate_alignment(predictions)
    
    # Optimize weights if requested
    if optimize_weights:
        print("\n--- Optimizing blending weights ---")
        
        # Try to load OOF predictions first (best option)
        oof_predictions = load_oof_predictions(cfg)
        
        if oof_predictions:
            print("Using OOF predictions for optimization (best practice)...")
            # Load true target values
            train = pd.read_csv(local_config.TRAIN_PROCESS8_CSV)
            y_true = train["logSP"].values
            
            # Use all models that have OOF predictions (regardless of initial weight)
            # This allows optimization to discover which models are best
            if oof_predictions:
                optimal_weights = optimize_blending_weights(
                    oof_predictions,
                    y_true,
                    initial_weights=cfg["weights"]
                )
                
                # Update config weights for all models with OOF predictions
                for name in oof_predictions.keys():
                    cfg["weights"][name] = optimal_weights.get(name, 0.0)
                # Set models without OOF to 0
                for name in cfg["weights"]:
                    if name not in oof_predictions:
                        cfg["weights"][name] = 0.0
                
                # Reload predictions with updated weights (to include newly activated models)
                predictions = load_predictions(cfg)
                validate_alignment(predictions)
            else:
                print("  Warning: No OOF predictions available. Using config weights.")
        else:
            print("OOF predictions not available.")
            print("  Note: For best results, run stacking model first to generate OOF predictions.")
            print("  Using config weights for now.")
            print("  To optimize weights, you need OOF predictions from cross-validation.")
    
    print(f"\nBlending {len(predictions)} models...")
    print(f"Final weights: {cfg['weights']}")
    
    # Blend predictions
    blend = blend_predictions(predictions, cfg["weights"])
    
    # Validate submission format and ID matching
    validate_submission_wrapper(blend, len(blend), "Blending", test_ids=blend["Id"])
    
    # Save blended file
    out_path = local_config.get_model_submission_path(cfg["submission_name"], cfg["submission_filename"])
    blend.to_csv(out_path, index=False)
    
    print(f"--- Blended predictions saved to: {out_path} ---")
    print(f"Prediction range: ${blend['SalePrice'].min():,.0f} - ${blend['SalePrice'].max():,.0f}")


if __name__ == "__main__":
    # Set optimize_weights=True to automatically find best weights
    main(optimize_weights=True)
