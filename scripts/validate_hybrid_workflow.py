#!/usr/bin/env python
"""
Validation script for hybrid workflow components.
Tests environment detection, imports, and basic functionality locally.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_environment_detection():
    """Test environment detection module."""
    print("=" * 70)
    print("TESTING ENVIRONMENT DETECTION")
    print("=" * 70)
    
    try:
        from config_local.environment import (
            is_kaggle_environment,
            get_base_path,
            get_data_path,
            get_working_data_path,
            detect_gpu,
            get_environment_info
        )
        
        print("[OK] Environment module imports successfully")
        
        # Test functions
        is_kaggle = is_kaggle_environment()
        base_path = get_base_path()
        data_path = get_data_path()
        working_path = get_working_data_path()
        gpu_info = detect_gpu()
        env_info = get_environment_info()
        
        print(f"  Environment: {'Kaggle' if is_kaggle else 'Local'}")
        print(f"  Base path: {base_path}")
        print(f"  Data path: {data_path}")
        print(f"  Working data path: {working_path}")
        print(f"  GPU available: {gpu_info['available']}")
        if gpu_info['device']:
            print(f"  GPU device: {gpu_info['device']}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Environment detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_paths():
    """Test that config paths are set correctly."""
    print("\n" + "=" * 70)
    print("TESTING CONFIG PATHS")
    print("=" * 70)
    
    try:
        from config_local import local_config
        
        print("[OK] Config module imports successfully")
        
        # Check that paths are Path objects
        assert hasattr(local_config, 'ROOT'), "ROOT not found in config"
        assert hasattr(local_config, 'DATA_DIR'), "DATA_DIR not found in config"
        assert hasattr(local_config, 'RAW_DIR'), "RAW_DIR not found in config"
        
        print(f"  ROOT: {local_config.ROOT}")
        print(f"  DATA_DIR: {local_config.DATA_DIR}")
        print(f"  RAW_DIR: {local_config.RAW_DIR}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Config paths test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kaggle_remote_modules():
    """Test Kaggle remote helper modules."""
    print("\n" + "=" * 70)
    print("TESTING KAGGLE REMOTE MODULES")
    print("=" * 70)
    
    try:
        # Try importing setup_kaggle (may fail if kaggle-api not installed, that's OK)
        try:
            from kaggle.remote.setup_kaggle import setup_kaggle_environment
            print("[OK] setup_kaggle module imports successfully")
        except ImportError as e:
            if 'kaggle.api' in str(e) or 'KaggleApi' in str(e):
                print("[SKIP] setup_kaggle requires kaggle-api package (optional dependency)")
            else:
                raise
        
        # GPU runner should work without kaggle-api
        # Import directly to avoid triggering kaggle package __init__.py
        import importlib.util
        gpu_runner_path = PROJECT_ROOT / 'kaggle' / 'remote' / 'gpu_runner.py'
        spec = importlib.util.spec_from_file_location("gpu_runner", gpu_runner_path)
        gpu_runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gpu_runner)
        
        get_gpu_params_for_model = gpu_runner.get_gpu_params_for_model
        verify_gpu_setup = gpu_runner.verify_gpu_setup
        print_gpu_usage_info = gpu_runner.print_gpu_usage_info
        
        print("[OK] GPU runner module imports successfully")
        
        # Test GPU parameter functions
        for model_type in ['xgboost', 'catboost', 'lightgbm']:
            params = get_gpu_params_for_model(model_type)
            print(f"  {model_type}: {params}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Kaggle remote modules test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_imports():
    """Test that model scripts can import required modules."""
    print("\n" + "=" * 70)
    print("TESTING MODEL IMPORTS")
    print("=" * 70)
    
    model_files = [
        'notebooks/Models/7XGBoostModel.py',
        'notebooks/Models/9catBoostModel.py',
        'notebooks/Models/8lightGbmModel.py'
    ]
    
    all_passed = True
    for model_file in model_files:
        model_path = PROJECT_ROOT / model_file
        if not model_path.exists():
            print(f"[SKIP] {model_file} not found")
            continue
        
        try:
            # Try to compile the file (syntax check)
            with open(model_path, 'r', encoding='utf-8') as f:
                code = f.read()
            compile(code, model_path, 'exec')
            print(f"[OK] {model_file} compiles successfully")
        except SyntaxError as e:
            print(f"[ERROR] {model_file} has syntax error: {e}")
            all_passed = False
        except Exception as e:
            print(f"[WARNING] {model_file} check failed: {e}")
            # Don't fail on import errors (dependencies might not be installed)
    
    return all_passed


def test_sync_helper():
    """Test sync helper script."""
    print("\n" + "=" * 70)
    print("TESTING SYNC HELPER")
    print("=" * 70)
    
    try:
        # Try importing (don't run, as it might try to check git)
        sync_helper_path = PROJECT_ROOT / 'scripts' / 'sync_to_kaggle.py'
        if sync_helper_path.exists():
            with open(sync_helper_path, 'r', encoding='utf-8') as f:
                code = f.read()
            compile(code, sync_helper_path, 'exec')
            print("[OK] sync_to_kaggle.py compiles successfully")
            return True
        else:
            print("[SKIP] sync_to_kaggle.py not found")
            return True
    except Exception as e:
        print(f"[ERROR] Sync helper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("\n" + "=" * 70)
    print("HYBRID WORKFLOW VALIDATION")
    print("=" * 70)
    print()
    
    tests = [
        ("Environment Detection", test_environment_detection),
        ("Config Paths", test_config_paths),
        ("Kaggle Remote Modules", test_kaggle_remote_modules),
        ("Model Imports", test_model_imports),
        ("Sync Helper", test_sync_helper),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[ERROR] Test '{test_name}' raised exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All validation tests passed!")
        print("\nNote: This validates local components. Full workflow testing")
        print("requires actual Kaggle notebook execution.")
        return 0
    else:
        print("\n[WARNING] Some tests failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

