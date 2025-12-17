"""Model blending script for ensemble predictions."""
import pandas as pd
from pathlib import Path
from typing import Dict

from config_local import local_config


def load_predictions() -> Dict[str, pd.DataFrame]:
    """Load all model predictions."""
    predictions = {}
    models = {
        "xgb": "xgboost_Model.csv",
        "lgb": "lightGBM_Model.csv",
        "cat": "catboost_Model.csv"
    }
    
    for name, filename in models.items():
        path = local_config.SUBMISSIONS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Prediction file not found: {path}")
        predictions[name] = pd.read_csv(path)
    
    return predictions


def validate_alignment(predictions: Dict[str, pd.DataFrame]) -> None:
    """Validate that all predictions have aligned Id columns."""
    ids = [pred["Id"] for pred in predictions.values()]
    for i, id_col in enumerate(ids[1:], start=1):
        if not ids[0].equals(id_col):
            raise ValueError(f"Id alignment mismatch between predictions {i} and 0")


def blend_predictions(
    predictions: Dict[str, pd.DataFrame],
    weights: Dict[str, float]
) -> pd.DataFrame:
    """
    Blend predictions using weighted average.
    
    Args:
        predictions: Dictionary mapping model names to DataFrames
        weights: Dictionary mapping model names to weights
        
    Returns:
        Blended predictions DataFrame
    """
    # Normalize weights
    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}
    
    # Initialize blended predictions
    blend = predictions["xgb"].copy()
    
    # Weighted average
    blend["SalePrice"] = sum(
        normalized_weights[name] * pred["SalePrice"]
        for name, pred in predictions.items()
    )
    
    return blend


def main() -> None:
    """Main entry point for blending."""
    # Load predictions
    print("📥 Loading predictions...")
    predictions = load_predictions()
    
    # Validate alignment
    print("✅ Validating alignment...")
    validate_alignment(predictions)
    
    # Blending weights
    weights = {
        "xgb": 2.0,
        "lgb": 0.5,
        "cat": 1.0
    }
    
    print(f"⚖️  Blending with weights: {weights}")
    
    # Blend predictions
    blend = blend_predictions(predictions, weights)
    
    # Save blended file
    output_path = local_config.SUBMISSIONS_DIR / "blend_xgb_lgb_cat_Model.csv"
    blend.to_csv(output_path, index=False)
    
    print(f"✅ Blended predictions saved to: {output_path}")
    print(f"📊 Prediction range: ${blend['SalePrice'].min():,.0f} - ${blend['SalePrice'].max():,.0f}")


if __name__ == "__main__":
    main()
