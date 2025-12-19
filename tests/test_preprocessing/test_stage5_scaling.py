"""Tests for Stage 5: Scaling"""
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
class TestStage5Scaling:
    """Tests for Stage 5: Scaling"""
    
    def test_stage5_output_exists(self):
        """Test that stage 5 output files exist."""
        assert Path(cfg.TRAIN_PROCESS5_CSV).exists(), "Train Process5 CSV not found"
        assert Path(cfg.TEST_PROCESS5_CSV).exists(), "Test Process5 CSV not found"
    
    def test_row_count_unchanged(self):
        """Test that row counts are unchanged."""
        train_stage4 = pd.read_csv(cfg.TRAIN_PROCESS4_CSV)
        train_stage5 = pd.read_csv(cfg.TRAIN_PROCESS5_CSV)
        test_stage4 = pd.read_csv(cfg.TEST_PROCESS4_CSV)
        test_stage5 = pd.read_csv(cfg.TEST_PROCESS5_CSV)
        
        assert len(train_stage4) == len(train_stage5), "Train row count should not change"
        assert len(test_stage4) == len(test_stage5), "Test row count should not change"
    
    def test_scaled_features_reasonable(self):
        """Test that scaled features have reasonable values."""
        train = pd.read_csv(cfg.TRAIN_PROCESS5_CSV)
        
        # Check numeric features (excluding target)
        numeric_cols = train.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != 'logSP']
        
        for col in numeric_cols[:10]:  # Check first 10
            values = train[col].values
            # Scaled values should be finite
            assert np.isfinite(values).all(), f"Column {col} has non-finite values"
    
    def test_comprehensive_validation(self):
        """Run comprehensive validation for stage 5."""
        results = validate_preprocessing_stage(5, stop_on_error=False)
        
        if results.get('skipped'):
            pytest.skip(results.get('reason', 'Stage 5 output not found'))
        
        assert results.get('passed', False), f"Validation failed: {results.get('errors', [])}"

