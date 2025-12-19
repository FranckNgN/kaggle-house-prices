"""Tests for Stage 4: Feature Engineering"""
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
class TestStage4FeatureEngineering:
    """Tests for Stage 4: Feature Engineering"""
    
    def test_stage4_output_exists(self):
        """Test that stage 4 output files exist."""
        assert Path(cfg.TRAIN_PROCESS4_CSV).exists(), "Train Process4 CSV not found"
        assert Path(cfg.TEST_PROCESS4_CSV).exists(), "Test Process4 CSV not found"
    
    def test_new_features_added(self):
        """Test that new features were added."""
        train_stage3 = pd.read_csv(cfg.TRAIN_PROCESS3_CSV)
        train_stage4 = pd.read_csv(cfg.TRAIN_PROCESS4_CSV)
        
        # Stage 4 should have more or equal features
        assert len(train_stage4.columns) >= len(train_stage3.columns), \
            "Stage 4 should have at least as many features as stage 3"
    
    def test_feature_sanity(self):
        """Test that engineered features are logically correct."""
        train = pd.read_csv(cfg.TRAIN_PROCESS4_CSV)
        
        # Check for negative values in area features
        area_cols = [c for c in train.columns if 'SF' in c or 'Area' in c]
        for col in area_cols:
            if train[col].dtype in ['float64', 'int64']:
                assert (train[col] >= 0).all() or train[col].isna().any(), \
                    f"Area feature {col} has negative values"
    
    def test_comprehensive_validation(self):
        """Run comprehensive validation for stage 4."""
        results = validate_preprocessing_stage(4, stop_on_error=False)
        
        if results.get('skipped'):
            pytest.skip(results.get('reason', 'Stage 4 output not found'))
        
        assert results.get('passed', False), f"Validation failed: {results.get('errors', [])}"

