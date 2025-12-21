#!/usr/bin/env python
"""Comprehensive test script for hybrid workflow components."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_environment_detection():
    """Test environment detection."""
    print("=" * 70)
    print("1. ENVIRONMENT DETECTION")
    print("=" * 70)
    try:
        from config_local.environment import (
            is_kaggle_environment,
            get_base_path,
            detect_gpu,
            get_environment_info
        )
        
        is_kaggle = is_kaggle_environment()
        base_path = get_base_path()
        gpu_info = detect_gpu()
        env_info = get_environment_info()
        
        print(f"[OK] Is Kaggle environment: {is_kaggle}")
        print(f"[OK] Base path: {base_path}")
        print(f"[OK] GPU available: {gpu_info['available']}")
        if gpu_info['device']:
            print(f"[OK] GPU device: {gpu_info['device']}")
        print(f"[OK] Data path: {env_info['data_path']}")
        print(f"[OK] Working path: {env_info['working_data_path']}")
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def test_gpu_runner():
    """Test GPU runner module."""
    print("\n" + "=" * 70)
    print("2. GPU RUNNER MODULE")
    print("=" * 70)
    try:
        from kaggle.remote.gpu_runner import get_gpu_params_for_model
        
        for model_type in ['xgboost', 'catboost', 'lightgbm']:
            params = get_gpu_params_for_model(model_type)
            print(f"[OK] {model_type}: {params}")
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_paths():
    """Test config paths."""
    print("\n" + "=" * 70)
    print("3. CONFIG PATHS")
    print("=" * 70)
    try:
        from config_local import local_config
        
        print(f"[OK] ROOT: {local_config.ROOT}")
        print(f"[OK] DATA_DIR: {local_config.DATA_DIR}")
        print(f"[OK] RAW_DIR: {local_config.RAW_DIR}")
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def test_model_scripts():
    """Test that model scripts have GPU detection."""
    print("\n" + "=" * 70)
    print("4. MODEL SCRIPTS GPU DETECTION")
    print("=" * 70)
    try:
        model_files = [
            'notebooks/Models/7XGBoostModel.py',
            'notebooks/Models/9catBoostModel.py',
            'notebooks/Models/8lightGbmModel.py'
        ]
        
        all_ok = True
        for model_file in model_files:
            model_path = PROJECT_ROOT / model_file
            if model_path.exists():
                content = model_path.read_text(encoding='utf-8')
                if 'from kaggle.remote.gpu_runner import get_gpu_params_for_model' in content:
                    print(f"[OK] {Path(model_file).name} has GPU detection")
                else:
                    print(f"[WARN] {Path(model_file).name} missing GPU import")
                    all_ok = False
            else:
                print(f"[SKIP] {Path(model_file).name} not found")
        
        return all_ok
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def test_kaggle_package():
    """Test kaggle package imports."""
    print("\n" + "=" * 70)
    print("5. KAGGLE PACKAGE IMPORTS")
    print("=" * 70)
    try:
        from kaggle import get_username, COMPETITION_NAME
        
        username = get_username()
        print(f"[OK] Username: {username}")
        print(f"[OK] Competition: {COMPETITION_NAME}")
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_setup_kaggle():
    """Test setup_kaggle module."""
    print("\n" + "=" * 70)
    print("6. SETUP KAGGLE MODULE")
    print("=" * 70)
    try:
        from kaggle.remote.setup_kaggle import setup_kaggle_environment
        
        print("[OK] setup_kaggle_environment imported successfully")
        # Don't actually call it (would print a lot), just test import
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("HYBRID WORKFLOW COMPREHENSIVE TEST")
    print("=" * 70)
    
    tests = [
        ("Environment Detection", test_environment_detection),
        ("GPU Runner", test_gpu_runner),
        ("Config Paths", test_config_paths),
        ("Model Scripts", test_model_scripts),
        ("Kaggle Package", test_kaggle_package),
        ("Setup Kaggle", test_setup_kaggle),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[ERROR] Test '{test_name}' raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed! Hybrid workflow is ready.")
        return 0
    else:
        print("\n[WARNING] Some tests failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

