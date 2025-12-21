"""Environment detection and path configuration for local and Kaggle environments."""
import os
import subprocess
from pathlib import Path
from typing import Optional

# Competition name for Kaggle
COMPETITION_NAME = "house-prices-advanced-regression-techniques"


def is_kaggle_environment() -> bool:
    """
    Detect if code is running in Kaggle environment.
    
    Checks for existence of /kaggle/working directory which is unique to Kaggle notebooks.
    
    Returns:
        True if running on Kaggle, False otherwise
    """
    return Path("/kaggle/working").exists()


def get_base_path() -> Path:
    """
    Get the base project path based on environment.
    
    Returns:
        Path to project root: /kaggle/working/project on Kaggle, 
        or local project root otherwise
    """
    if is_kaggle_environment():
        # On Kaggle, project is typically cloned to /kaggle/working/project
        kaggle_project = Path("/kaggle/working/project")
        if kaggle_project.exists():
            return kaggle_project
        
        # Alternative: if running from within a cloned repo in /kaggle/working
        # Try to find the project root by looking for config_local directory
        current = Path("/kaggle/working")
        for potential_root in [current, current / "house-prices-starter"]:
            if (potential_root / "config_local").exists():
                return potential_root
        
        # Fallback: use /kaggle/working
        return Path("/kaggle/working")
    else:
        # Local environment: return project root (parent of config_local)
        return Path(__file__).resolve().parents[1]


def get_data_path() -> Path:
    """
    Get the data directory path based on environment.
    
    Returns:
        Path to data directory:
        - Kaggle: /kaggle/input/competition-name/ (competition data)
        - Local: data/ directory relative to project root
    """
    if is_kaggle_environment():
        return Path(f"/kaggle/input/{COMPETITION_NAME}")
    else:
        return get_base_path() / "data"


def get_working_data_path() -> Path:
    """
    Get the working data directory for outputs and intermediate files.
    
    Returns:
        Path to working data directory:
        - Kaggle: /kaggle/working/data/ (for outputs)
        - Local: data/ directory relative to project root
    """
    if is_kaggle_environment():
        return Path("/kaggle/working/data")
    else:
        return get_base_path() / "data"


def detect_gpu() -> dict:
    """
    Detect GPU availability and return GPU information.
    
    Returns:
        Dictionary with:
        - 'available': bool - Whether GPU is available
        - 'device': str - GPU device name if available
        - 'method': str - How GPU was detected
    """
    result = {
        'available': False,
        'device': None,
        'method': None
    }
    
    # Method 1: Check nvidia-smi (most reliable)
    try:
        nvidia_result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if nvidia_result.returncode == 0 and nvidia_result.stdout.strip():
            result['available'] = True
            result['device'] = nvidia_result.stdout.strip().split('\n')[0]
            result['method'] = 'nvidia-smi'
            return result
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    # Method 2: Check CUDA_VISIBLE_DEVICES environment variable
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices and cuda_devices != '-1':
        result['available'] = True
        result['device'] = f"CUDA device {cuda_devices}"
        result['method'] = 'CUDA_VISIBLE_DEVICES'
        return result
    
    # Method 3: Try PyTorch (if available)
    try:
        import torch
        if torch.cuda.is_available():
            result['available'] = True
            result['device'] = torch.cuda.get_device_name(0)
            result['method'] = 'pytorch'
            return result
    except ImportError:
        pass
    
    # Method 4: Try TensorFlow (if available)
    try:
        import tensorflow as tf
        if tf.test.is_gpu_available():
            result['available'] = True
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                result['device'] = str(gpus[0])
                result['method'] = 'tensorflow'
                return result
    except ImportError:
        pass
    
    return result


def get_kaggle_input_path(file_path: str) -> Path:
    """
    Convert local file path to Kaggle input path.
    
    Maps local paths like 'data/raw/train.csv' to 
    '/kaggle/input/competition-name/train.csv'
    
    Args:
        file_path: Local file path relative to data directory
        
    Returns:
        Path object for Kaggle input or local path
    """
    if is_kaggle_environment():
        # Extract filename from path
        filename = Path(file_path).name
        return Path(f"/kaggle/input/{COMPETITION_NAME}") / filename
    else:
        return get_base_path() / file_path


def get_kaggle_working_path(file_path: str) -> Path:
    """
    Convert local file path to Kaggle working path.
    
    Maps local paths to /kaggle/working/ paths on Kaggle,
    or keeps local paths when running locally.
    
    Args:
        file_path: Local file path relative to project root
        
    Returns:
        Path object for Kaggle working or local path
    """
    if is_kaggle_environment():
        # Ensure path starts with /kaggle/working
        if file_path.startswith('/'):
            # Already absolute, use as-is if in /kaggle/working
            return Path(file_path)
        else:
            # Relative path, prepend /kaggle/working
            return Path("/kaggle/working") / file_path
    else:
        return get_base_path() / file_path


def setup_kaggle_symlinks(base_path: Optional[Path] = None) -> None:
    """
    Create symlinks from project data/raw/ to Kaggle input for compatibility.
    
    This allows code to use local paths even on Kaggle by creating symlinks.
    
    Args:
        base_path: Project base path (defaults to get_base_path())
    """
    if not is_kaggle_environment():
        return  # Only needed on Kaggle
    
    if base_path is None:
        base_path = get_base_path()
    
    raw_dir = base_path / "data" / "raw"
    kaggle_input = Path(f"/kaggle/input/{COMPETITION_NAME}")
    
    if not kaggle_input.exists():
        print(f"[WARNING] Kaggle input directory not found: {kaggle_input}")
        print("         Make sure you've added the competition dataset to your notebook.")
        return
    
    # Create data/raw directory if it doesn't exist
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Create symlinks for competition files
    competition_files = ['train.csv', 'test.csv', 'data_description.txt', 'sample_submission.csv']
    
    for filename in competition_files:
        source = kaggle_input / filename
        target = raw_dir / filename
        
        if source.exists() and not target.exists():
            try:
                target.symlink_to(source)
                print(f"[OK] Created symlink: {target} -> {source}")
            except Exception as e:
                print(f"[WARNING] Failed to create symlink {target}: {e}")
        elif target.exists():
            # Check if it's already a symlink or real file
            if target.is_symlink():
                print(f"[INFO] Symlink already exists: {target}")
            else:
                print(f"[INFO] File already exists (not symlink): {target}")


def get_environment_info() -> dict:
    """
    Get comprehensive environment information.
    
    Returns:
        Dictionary with environment details
    """
    base_path = get_base_path()
    gpu_info = detect_gpu()
    
    return {
        'is_kaggle': is_kaggle_environment(),
        'base_path': str(base_path),
        'data_path': str(get_data_path()),
        'working_data_path': str(get_working_data_path()),
        'gpu_available': gpu_info['available'],
        'gpu_device': gpu_info['device'],
        'gpu_method': gpu_info['method'],
        'competition_name': COMPETITION_NAME
    }


if __name__ == "__main__":
    # Print environment information when run directly
    info = get_environment_info()
    print("=" * 70)
    print("ENVIRONMENT INFORMATION")
    print("=" * 70)
    print(f"Environment: {'Kaggle' if info['is_kaggle'] else 'Local'}")
    print(f"Base Path: {info['base_path']}")
    print(f"Data Path: {info['data_path']}")
    print(f"Working Data Path: {info['working_data_path']}")
    print(f"GPU Available: {info['gpu_available']}")
    if info['gpu_device']:
        print(f"GPU Device: {info['gpu_device']} (detected via {info['gpu_method']})")
    print(f"Competition: {info['competition_name']}")
    print("=" * 70)

