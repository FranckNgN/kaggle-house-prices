"""Base tests for model validation."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import config_local.local_config as cfg
    HAVE_CFG = True
except ImportError:
    HAVE_CFG = False


@pytest.mark.integration
@pytest.mark.skipif(not HAVE_CFG, reason="config_local not available")
class TestModelBase:
    """Base tests for model validation."""
    
    @pytest.fixture
    def processed_data(self):
        """Load processed data for model testing."""
        if not Path(cfg.TRAIN_PROCESS6_CSV).exists():
            pytest.skip("Processed data not found")
        
        train = pd.read_csv(cfg.TRAIN_PROCESS6_CSV)
        test = pd.read_csv(cfg.TEST_PROCESS6_CSV)
        
        y = train['logSP']
        X_train = train.drop(columns=['logSP'])
        X_test = test
        
        return X_train, y, X_test
    
    def test_data_loaded_correctly(self, processed_data):
        """Test that data is loaded correctly for modeling."""
        X_train, y, X_test = processed_data
        
        assert len(X_train) > 0, "X_train is empty"
        assert len(y) > 0, "y is empty"
        assert len(X_test) > 0, "X_test is empty"
        assert len(X_train) == len(y), "X_train and y have different lengths"
    
    def test_feature_matrices_have_same_features(self, processed_data):
        """Test that train and test have the same features."""
        X_train, y, X_test = processed_data
        
        train_cols = set(X_train.columns)
        test_cols = set(X_test.columns)
        
        assert train_cols == test_cols, \
            f"Feature mismatch: train_only={train_cols-test_cols}, test_only={test_cols-train_cols}"
    
    def test_no_nans_in_features(self, processed_data):
        """Test that feature matrices have no NaNs."""
        X_train, y, X_test = processed_data
        
        assert X_train.isnull().sum().sum() == 0, "X_train has missing values"
        assert X_test.isnull().sum().sum() == 0, "X_test has missing values"
        assert y.isnull().sum() == 0, "y has missing values"
    
    def test_target_in_reasonable_range(self, processed_data):
        """Test that target values are in reasonable range."""
        X_train, y, X_test = processed_data
        
        # logSP should be positive and reasonable
        assert (y > 0).all(), "Target has non-positive values"
        assert (y < 20).all(), "Target values seem too large (log scale)"
        assert y.mean() > 0, "Target mean should be positive"

