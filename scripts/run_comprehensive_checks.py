#!/usr/bin/env python
"""Comprehensive sanity checks for all preprocessing stages."""
import pandas as pd
import numpy as np
from pathlib import Path
from config_local import local_config
from utils.checks import validate_preprocessing_stage

def main():
    print("="*70)
    print("COMPREHENSIVE SANITY CHECK - ALL PREPROCESSING STAGES")
    print("="*70)
    
    # Check all stages
    all_passed = True
    for stage in range(1, 9):
        print(f"\n{'='*70}")
        print(f"CHECKING STAGE {stage}")
        print(f"{'='*70}")
        try:
            result = validate_preprocessing_stage(stage, stop_on_error=False)
            if result.get('passed', False):
                print(f"[PASS] Stage {stage}: ALL CHECKS PASSED")
            else:
                print(f"[FAIL] Stage {stage}: FAILED")
                print(f"   Errors: {result.get('errors', [])}")
                all_passed = False
        except Exception as e:
            print(f"[ERROR] Stage {stage}: ERROR - {e}")
            all_passed = False
    
    # Final comprehensive check
    print(f"\n{'='*70}")
    print("FINAL COMPREHENSIVE CHECK - STAGE 8")
    print(f"{'='*70}")
    
    train = pd.read_csv(local_config.TRAIN_PROCESS8_CSV)
    test = pd.read_csv(local_config.TEST_PROCESS8_CSV)
    
    print("\n1. DATA INTEGRITY:")
    print(f"   Train samples: {len(train)}")
    print(f"   Test samples: {len(test)}")
    print(f"   Train features: {len(train.columns)}")
    print(f"   Test features: {len(test.columns)}")
    print(f"   Target present: {'logSP' in train.columns}")
    if 'logSP' in train.columns:
        print(f"   Target range: {train['logSP'].min():.4f} to {train['logSP'].max():.4f}")
        print(f"   Target mean: {train['logSP'].mean():.4f}")
    
    print("\n2. MISSING VALUES:")
    train_missing = train.isnull().sum().sum()
    test_missing = test.isnull().sum().sum()
    print(f"   Train missing: {train_missing} {'[OK]' if train_missing == 0 else '[FAIL]'}")
    print(f"   Test missing: {test_missing} {'[OK]' if test_missing == 0 else '[FAIL]'}")
    
    print("\n3. NEW ERROR-DRIVEN FEATURES:")
    new_features = ['Qual_Age_Interaction', 'RemodAge_FromBuild', 'Is_Remodeled', 'OverallQual_Squared']
    for feat in new_features:
        if feat in train.columns:
            print(f"   {feat}: [OK] Present (range: {train[feat].min():.2f} to {train[feat].max():.2f})")
        else:
            print(f"   {feat}: [FAIL] MISSING")
            all_passed = False
    
    print("\n4. COLUMN PARITY:")
    train_cols = set(train.columns) - {'logSP'}
    test_cols = set(test.columns)
    shared = train_cols & test_cols
    train_only = train_cols - test_cols
    test_only = test_cols - train_cols
    print(f"   Shared features: {len(shared)} {'[OK]' if len(shared) > 0 else '[FAIL]'}")
    if train_only:
        print(f"   ⚠️  Train-only: {train_only}")
    if test_only:
        print(f"   ⚠️  Test-only: {test_only}")
    
    print("\n5. FEATURE STATISTICS:")
    print(f"   Numeric features: {len(train.select_dtypes(include=[np.number]).columns)}")
    print(f"   All features are numeric: {train.select_dtypes(exclude=[np.number]).empty}")
    
    print("\n6. TARGET VALIDATION:")
    if 'logSP' in train.columns:
        target = train['logSP']
        print(f"   No NaN: {target.notna().all()} {'[OK]' if target.notna().all() else '[FAIL]'}")
        print(f"   No Inf: {np.isfinite(target).all()} {'[OK]' if np.isfinite(target).all() else '[FAIL]'}")
        print(f"   Reasonable range: {target.min() > 0 and target.max() < 20} {'[OK]' if target.min() > 0 and target.max() < 20 else '[FAIL]'}")
    
    print(f"\n{'='*70}")
    if all_passed:
        print("[PASS] ALL SANITY CHECKS PASSED")
    else:
        print("[FAIL] SOME CHECKS FAILED - REVIEW ABOVE")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()

