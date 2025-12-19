"""Tests for Stage 1: Data Cleaning"""
import pytest
import pandas as pd
from pathlib import Path
from utils.checks import (
    validate_preprocessing_stage,
    check_raw_data_integrity,
    DataLeakageError,
    DataIntegrityError
)

try:
    import config_local.local_config as cfg
    HAVE_CFG = True
except ImportError:
    HAVE_CFG = False


@pytest.mark.integration
@pytest.mark.skipif(not HAVE_CFG, reason="config_local not available")
class TestStage1Cleaning:
    """Tests for Stage 1: Data Cleaning"""
    
    def test_raw_data_integrity(self):
        """Test that raw data has basic integrity."""
        if not Path(cfg.TRAIN_CSV).exists():
            pytest.skip("Raw train data not found")
        
        results = check_raw_data_integrity(
            Path(cfg.TRAIN_CSV),
            Path(cfg.TEST_CSV)
        )
        
        assert results.get('train_has_rows', False), "Train has no rows"
        assert results.get('test_has_rows', False), "Test has no rows"
        assert results.get('target_exists', False), "Target column missing"
        assert results.get('target_all_positive', False), "Target has non-positive values"
        assert results.get('target_no_nans', False), "Target has NaNs"
    
    def test_stage1_output_exists(self):
        """Test that stage 1 output files exist."""
        assert Path(cfg.TRAIN_PROCESS1_CSV).exists(), "Train Process1 CSV not found"
        assert Path(cfg.TEST_PROCESS1_CSV).exists(), "Test Process1 CSV not found"
    
    def test_no_missing_values_after_cleaning(self):
        """Test that missing values are filled after cleaning."""
        train = pd.read_csv(cfg.TRAIN_PROCESS1_CSV)
        test = pd.read_csv(cfg.TEST_PROCESS1_CSV)
        
        # Test should have no NaNs (except maybe target in train)
        assert test.isnull().sum().sum() == 0, "Test has missing values after cleaning"
        
        # Train should have no NaNs except possibly in target (which should be handled)
        train_nulls = train.drop(columns=['SalePrice'], errors='ignore').isnull().sum().sum()
        assert train_nulls == 0, f"Train has {train_nulls} missing values in features"
    
    def test_no_train_test_leakage(self):
        """Test that train and test are independent."""
        train = pd.read_csv(cfg.TRAIN_PROCESS1_CSV)
        test = pd.read_csv(cfg.TEST_PROCESS1_CSV)
        
        if 'Id' in train.columns and 'Id' in test.columns:
            train_ids = set(train['Id'].values)
            test_ids = set(test['Id'].values)
            assert len(train_ids & test_ids) == 0, "Train and test share IDs"
    
    def test_column_parity(self):
        """Test that train and test have same columns (except target)."""
        train = pd.read_csv(cfg.TRAIN_PROCESS1_CSV)
        test = pd.read_csv(cfg.TEST_PROCESS1_CSV)
        
        train_cols = set(train.columns) - {'SalePrice'}
        test_cols = set(test.columns)
        
        assert train_cols == test_cols, f"Column mismatch: train={train_cols-test_cols}, test={test_cols-train_cols}"
    
    def test_comprehensive_validation(self):
        """Run comprehensive validation for stage 1."""
        results = validate_preprocessing_stage(1, stop_on_error=False)
        
        if results.get('skipped'):
            pytest.skip(results.get('reason', 'Stage 1 output not found'))
        
        assert results.get('passed', False), f"Validation failed: {results.get('errors', [])}"
        assert results.get('no_leakage', True), f"Leakage detected: {results.get('no_leakage_error', '')}"

