#!/usr/bin/env python
"""
Run preprocessing pipeline up to stage 3 (skew/kurtosis) but skip stage 4 (feature engineering).
This gives us raw preprocessing without feature engineering.
"""
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_local import local_config

def run_preprocessing_stage(stage_num: int):
    """Run a single preprocessing stage."""
    stage_file = PROJECT_ROOT / "notebooks" / "preprocessing" / f"{stage_num}*.py"
    
    # Find the actual file
    import glob
    files = glob.glob(str(stage_file))
    if not files:
        print(f"[ERROR] Stage {stage_num} file not found")
        return False
    
    stage_path = Path(files[0])
    print(f"\n{'='*70}")
    print(f"Running Stage {stage_num}: {stage_path.name}")
    print(f"{'='*70}")
    
    # Use run_with_venv to ensure proper environment
    try:
        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        
        result = subprocess.run(
            [sys.executable, "scripts/run_with_venv.py", str(stage_path)],
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=False,
            env=env
        )
        print(f"[OK] Stage {stage_num} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Stage {stage_num} failed")
        return False

def main():
    print("=" * 70)
    print("RAW PREPROCESSING (Without Feature Engineering)")
    print("=" * 70)
    print("\nThis will run:")
    print("  Stage 1: Cleaning")
    print("  Stage 2: Data Engineering (basic transforms)")
    print("  Stage 3: Skew/Kurtosis handling")
    print("  [SKIP] Stage 4: Feature Engineering")
    print("  Stage 5: Scaling")
    print("  Stage 6: Categorical Encoding")
    print("  Stage 7: Feature Selection")
    print("  Stage 8: Target Encoding")
    print("\n" + "=" * 70)
    
    # Run stages 1-3
    for stage in [1, 2, 3]:
        if not run_preprocessing_stage(stage):
            print(f"\n[ERROR] Preprocessing failed at stage {stage}")
            sys.exit(1)
    
    # Skip stage 4 (feature engineering)
    print(f"\n{'='*70}")
    print("SKIPPING Stage 4: Feature Engineering")
    print(f"{'='*70}")
    
    # Copy process3 to process4 (since we're skipping feature engineering)
    import pandas as pd
    
    print("\nCopying process3 outputs to process4 (skipping feature engineering)...")
    train_process3 = pd.read_csv(local_config.TRAIN_PROCESS3_CSV)
    test_process3 = pd.read_csv(local_config.TEST_PROCESS3_CSV)
    
    train_process3.to_csv(local_config.TRAIN_PROCESS4_CSV, index=False)
    test_process3.to_csv(local_config.TEST_PROCESS4_CSV, index=False)
    print("[OK] Process4 files created (same as process3)")
    
    # Continue with stages 5-8
    for stage in [5, 6, 7, 8]:
        if not run_preprocessing_stage(stage):
            print(f"\n[ERROR] Preprocessing failed at stage {stage}")
            sys.exit(1)
    
    print("\n" + "=" * 70)
    print("[SUCCESS] RAW PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  Train: {local_config.TRAIN_PROCESS8_CSV}")
    print(f"  Test: {local_config.TEST_PROCESS8_CSV}")
    
    # Check feature count
    train = pd.read_csv(local_config.TRAIN_PROCESS8_CSV)
    print(f"\nFeature count: {train.shape[1] - 1} (excluding target)")

if __name__ == "__main__":
    main()

