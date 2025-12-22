"""GPU runner utility for executing models on Kaggle."""
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config_local.environment import is_kaggle_environment, detect_gpu, get_environment_info


def get_gpu_params_for_model(model_type: str) -> dict:
    """
    Get GPU parameters for a specific model type.
    
    Args:
        model_type: Type of model ('xgboost', 'catboost', 'lightgbm')
        
    Returns:
        Dictionary of parameters to use GPU if available, CPU otherwise
    """
    gpu_info = detect_gpu()
    params = {}
    
    if not gpu_info['available']:
        print(f"[INFO] GPU not available. {model_type} will use CPU.")
        if model_type == 'xgboost':
            params['tree_method'] = 'hist'  # CPU method
        elif model_type == 'catboost':
            params['task_type'] = 'CPU'
        elif model_type == 'lightgbm':
            params['device'] = 'cpu'
        return params
    
    # GPU is available, set GPU parameters
    print(f"[INFO] GPU available. {model_type} will use GPU.")
    if model_type == 'xgboost':
        # Use 'hist' with device='cuda' instead of 'gpu_hist' for compatibility
        params['tree_method'] = 'hist'
        params['device'] = 'cuda'
    elif model_type == 'catboost':
        params['task_type'] = 'GPU'
    elif model_type == 'lightgbm':
        params['device'] = 'gpu'
    
    return params


def verify_gpu_setup() -> bool:
    """
    Verify GPU setup and print information.
    
    Returns:
        True if GPU is available, False otherwise
    """
    print("=" * 70)
    print("GPU SETUP VERIFICATION")
    print("=" * 70)
    
    if not is_kaggle_environment():
        print("[INFO] Not running on Kaggle. GPU verification skipped.")
        return False
    
    gpu_info = detect_gpu()
    env_info = get_environment_info()
    
    print(f"Environment: {'Kaggle' if env_info['is_kaggle'] else 'Local'}")
    print(f"GPU Available: {gpu_info['available']}")
    
    if gpu_info['available']:
        print(f"GPU Device: {gpu_info['device']}")
        print(f"Detection Method: {gpu_info['method']}")
        print()
        print("[OK] GPU is ready for model training")
        return True
    else:
        print()
        print("[WARNING] GPU is not available.")
        print("          Make sure GPU accelerator is enabled in notebook settings.")
        print("          Models will run on CPU (slower).")
        return False


def print_gpu_usage_info():
    """Print information about GPU usage for different models."""
    gpu_info = detect_gpu()
    
    print("=" * 70)
    print("GPU USAGE INFORMATION")
    print("=" * 70)
    
    if gpu_info['available']:
        print("[OK] GPU is available and will be used by:")
        print("  - XGBoost: tree_method='gpu_hist'")
        print("  - CatBoost: task_type='GPU'")
        print("  - LightGBM: device='gpu'")
    else:
        print("[INFO] GPU not available. Models will use CPU:")
        print("  - XGBoost: tree_method='hist'")
        print("  - CatBoost: task_type='CPU'")
        print("  - LightGBM: device='cpu'")
    
    print("=" * 70)


if __name__ == "__main__":
    verify_gpu_setup()
    print()
    print_gpu_usage_info()

