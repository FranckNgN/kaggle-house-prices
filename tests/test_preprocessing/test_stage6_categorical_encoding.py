"""Tests for Stage 6: Categorical Encoding"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from utils.checks import validate_preprocessing_stage

try:
    import config_local.local_config as cfg
    HAVE_CFG = True
except ImportError:
    HAVE_CFG = False


@pytest.mark.integration
@pytest.mark.skipif(not HAVE_CFG, reason="config_local not available")
class TestStage6CategoricalEncoding:
    """Tests for Stage 6: Categorical Encoding"""
    
    def test_stage6_output_exists(self):
        """Test that stage 6 output files exist."""
        assert Path(cfg.TRAIN_PROCESS6_CSV).exists(), "Train Process6 CSV not found"
        assert Path(cfg.TEST_PROCESS6_CSV).exists(), "Test Process6 CSV not found"
    
    def test_all_numeric_after_encoding(self):
        """Test that all features are numeric after encoding."""
        train = pd.read_csv(cfg.TRAIN_PROCESS6_CSV)
        test = pd.read_csv(cfg.TEST_PROCESS6_CSV)
        
        # Exclude target
        train_features = train.drop(columns=['logSP'], errors='ignore')
        test_features = test
        
        # All should be numeric
        assert train_features.select_dtypes(exclude=[np.number]).empty, \
            "Train should have only numeric features after encoding"
        assert test_features.select_dtypes(exclude=[np.number]).empty, \
            "Test should have only numeric features after encoding"
    
    def test_column_parity_final(self):
        """Test that train and test have same features (except target)."""
        train = pd.read_csv(cfg.TRAIN_PROCESS6_CSV)
        test = pd.read_csv(cfg.TEST_PROCESS6_CSV)
        
        train_cols = set(train.columns) - {'logSP'}
        test_cols = set(test.columns)
        
        assert train_cols == test_cols, \
            f"Final column mismatch: train_only={train_cols-test_cols}, test_only={test_cols-train_cols}"
    
    def test_target_preserved(self):
        """Test that target is preserved correctly."""
        train = pd.read_csv(cfg.TRAIN_PROCESS6_CSV)
        assert 'logSP' in train.columns, "logSP should be in final train data"
        assert train['logSP'].notna().all(), "logSP should have no missing values"
    
    def test_comprehensive_validation(self):
        """Run comprehensive validation for stage 6."""
        results = validate_preprocessing_stage(6, stop_on_error=False)
        
        if results.get('skipped'):
            pytest.skip(results.get('reason', 'Stage 6 output not found'))
        
        assert results.get('passed', False), f"Validation failed: {results.get('errors', [])}"

