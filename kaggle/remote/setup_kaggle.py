"""Setup script for Kaggle environment."""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config_local.environment import (
    is_kaggle_environment,
    get_base_path,
    setup_kaggle_symlinks,
    detect_gpu,
    get_environment_info
)


def setup_kaggle_environment():
    """
    Setup Kaggle environment for running the project.
    
    This function:
    1. Verifies we're running on Kaggle
    2. Sets up symlinks from project data/raw to Kaggle input
    3. Creates necessary directories
    4. Reports GPU availability
    """
    print("=" * 70)
    print("KAGGLE ENVIRONMENT SETUP")
    print("=" * 70)
    
    if not is_kaggle_environment():
        print("[WARNING] Not running on Kaggle. This script is for Kaggle notebooks only.")
        print("          On local machine, use standard setup procedures.")
        return False
    
    print("[OK] Running on Kaggle environment")
    print()
    
    # Get environment info
    base_path = get_base_path()
    print(f"Project root: {base_path}")
    print()
    
    # Setup symlinks
    print("Setting up data symlinks...")
    setup_kaggle_symlinks(base_path)
    print()
    
    # Check GPU
    print("Checking GPU availability...")
    gpu_info = detect_gpu()
    if gpu_info['available']:
        print(f"[OK] GPU available: {gpu_info['device']}")
        print(f"      Detected via: {gpu_info['method']}")
    else:
        print("[WARNING] GPU not available. Models will run on CPU.")
        print("          Make sure GPU accelerator is enabled in notebook settings.")
    print()
    
    # Print full environment info
    env_info = get_environment_info()
    print("Environment Information:")
    print(f"  Base Path: {env_info['base_path']}")
    print(f"  Data Path: {env_info['data_path']}")
    print(f"  Working Data Path: {env_info['working_data_path']}")
    print(f"  GPU Available: {env_info['gpu_available']}")
    if env_info['gpu_device']:
        print(f"  GPU Device: {env_info['gpu_device']}")
    print()
    
    print("=" * 70)
    print("[SUCCESS] Kaggle environment setup complete!")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    setup_kaggle_environment()

