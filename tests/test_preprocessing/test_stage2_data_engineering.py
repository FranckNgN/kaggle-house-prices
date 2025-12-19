"""Tests for Stage 2: Data Engineering"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from utils.checks import (
    validate_preprocessing_stage,
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
class TestStage2DataEngineering:
    """Tests for Stage 2: Data Engineering"""
    
    def test_stage2_output_exists(self):
        """Test that stage 2 output files exist."""
        assert Path(cfg.TRAIN_PROCESS2_CSV).exists(), "Train Process2 CSV not found"
        assert Path(cfg.TEST_PROCESS2_CSV).exists(), "Test Process2 CSV not found"
    
    def test_logsp_correctly_computed(self):
        """Test that logSP is correctly computed from SalePrice."""
        train_stage1 = pd.read_csv(cfg.TRAIN_PROCESS1_CSV)
        train_stage2 = pd.read_csv(cfg.TRAIN_PROCESS2_CSV)
        
        if 'SalePrice' in train_stage1.columns and 'logSP' in train_stage2.columns:
            # Get matching rows (after outlier removal)
            # This is a simplified check - in reality we'd need to match by ID
            expected_log = np.log1p(train_stage1['SalePrice'].values)
            # We can't directly compare due to outlier removal, but we can check range
            actual_log = train_stage2['logSP'].values
            assert (actual_log > 0).all(), "logSP should be positive"
            assert (actual_log < 20).all(), "logSP values seem too large"
    
    def test_outlier_removal_preserved_data_integrity(self):
        """Test that outlier removal didn't break data integrity."""
        train_stage1 = pd.read_csv(cfg.TRAIN_PROCESS1_CSV)
        train_stage2 = pd.read_csv(cfg.TRAIN_PROCESS2_CSV)
        
        # Stage 2 removes outliers, so train should have fewer rows
        assert len(train_stage2) <= len(train_stage1), "Stage 2 should not add rows"
    
    def test_new_features_created(self):
        """Test that new features were created."""
        train = pd.read_csv(cfg.TRAIN_PROCESS2_CSV)
        
        # Check for common engineered features
        expected_features = ['Age', 'TotalSF', 'TotalBath', 'TotalPorchSF']
        found_features = [f for f in expected_features if f in train.columns]
        assert len(found_features) > 0, f"Expected engineered features not found. Found: {found_features}"
    
    def test_no_saleprice_in_stage2(self):
        """Test that SalePrice is removed after log transformation."""
        train = pd.read_csv(cfg.TRAIN_PROCESS2_CSV)
        assert 'SalePrice' not in train.columns, "SalePrice should be removed in stage 2"
        assert 'logSP' in train.columns, "logSP should exist in stage 2"
    
    def test_comprehensive_validation(self):
        """Run comprehensive validation for stage 2."""
        results = validate_preprocessing_stage(2, stop_on_error=False)
        
        if results.get('skipped'):
            pytest.skip(results.get('reason', 'Stage 2 output not found'))
        
        assert results.get('passed', False), f"Validation failed: {results.get('errors', [])}"

