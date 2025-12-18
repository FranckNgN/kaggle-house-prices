"""Model blending script for ensemble predictions."""
import pandas as pd
from pathlib import Path
from typing import Dict

from config_local import local_config, model_config


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


def main() -> None:
    """Main entry point for blending."""
    cfg = model_config.BLENDING
    
    # Load predictions
    print("--- Loading predictions ---")
    predictions = load_predictions(cfg)
    
    # Validate alignment
    print("--- Validating alignment ---")
    validate_alignment(predictions)
    
    print(f"Blending {len(predictions)} models...")
    
    # Blend predictions
    blend = blend_predictions(predictions, cfg["weights"])
    
    # Save blended file
    out_path = local_config.get_model_submission_path(cfg["submission_name"], cfg["submission_filename"])
    blend.to_csv(out_path, index=False)
    
    print(f"--- Blended predictions saved to: {output_path} ---")
    print(f"Prediction range: ${blend['SalePrice'].min():,.0f} - ${blend['SalePrice'].max():,.0f}")


if __name__ == "__main__":
    main()
