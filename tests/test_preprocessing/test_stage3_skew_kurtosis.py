"""Tests for Stage 3: Skewness and Kurtosis Handling"""
import pytest
import pandas as pd
from pathlib import Path
from utils.checks import validate_preprocessing_stage

try:
    import config_local.local_config as cfg
    HAVE_CFG = True
except ImportError:
    HAVE_CFG = False


@pytest.mark.integration
@pytest.mark.skipif(not HAVE_CFG, reason="config_local not available")
class TestStage3SkewKurtosis:
    """Tests for Stage 3: Skewness and Kurtosis Handling"""
    
    def test_stage3_output_exists(self):
        """Test that stage 3 output files exist."""
        assert Path(cfg.TRAIN_PROCESS3_CSV).exists(), "Train Process3 CSV not found"
        assert Path(cfg.TEST_PROCESS3_CSV).exists(), "Test Process3 CSV not found"
    
    def test_row_count_unchanged(self):
        """Test that row counts are unchanged in stage 3."""
        train_stage2 = pd.read_csv(cfg.TRAIN_PROCESS2_CSV)
        train_stage3 = pd.read_csv(cfg.TRAIN_PROCESS3_CSV)
        
        assert len(train_stage2) == len(train_stage3), "Row count should not change in stage 3"
    
    def test_target_preserved(self):
        """Test that target column is preserved."""
        train = pd.read_csv(cfg.TRAIN_PROCESS3_CSV)
        assert 'logSP' in train.columns, "logSP should be preserved in stage 3"
    
    def test_comprehensive_validation(self):
        """Run comprehensive validation for stage 3."""
        results = validate_preprocessing_stage(3, stop_on_error=False)
        
        if results.get('skipped'):
            pytest.skip(results.get('reason', 'Stage 3 output not found'))
        
        assert results.get('passed', False), f"Validation failed: {results.get('errors', [])}"

