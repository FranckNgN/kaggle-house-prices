"""Tests to detect data leakage in model training."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from utils.checks import (
    check_model_no_target_leakage,
    check_cv_properly_implemented,
    check_predictions_sanity,
    check_submission_format,
    DataLeakageError
)

try:
    import config_local.local_config as cfg
    HAVE_CFG = True
except ImportError:
    HAVE_CFG = False


@pytest.mark.integration
@pytest.mark.skipif(not HAVE_CFG, reason="config_local not available")
class TestModelLeakage:
    """Tests to detect data leakage in model training."""
    
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
    
    def test_target_not_in_features(self, processed_data):
        """Test that target is not accidentally included in features."""
        X_train, y, X_test = processed_data
        
        try:
            check_model_no_target_leakage(X_train, y, X_test, "Test Model")
        except DataLeakageError as e:
            pytest.fail(f"Target leakage detected: {e}")
    
    def test_cv_splits_valid(self):
        """Test that CV splits are properly implemented."""
        # Create a simple CV scenario
        n_samples = 100
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        X_dummy = np.random.randn(n_samples, 10)
        
        cv_splits = list(kf.split(X_dummy))
        
        try:
            check_cv_properly_implemented(cv_splits, n_samples)
        except DataLeakageError as e:
            pytest.fail(f"CV implementation issue: {e}")
    
    def test_predictions_sanity(self):
        """Test that model predictions are reasonable."""
        # Create dummy predictions
        predictions_log = np.random.normal(12, 1, 100)  # Reasonable log scale
        
        try:
            check_predictions_sanity(predictions_log, "Test Model", target_is_log=True)
        except Exception as e:
            pytest.fail(f"Prediction sanity check failed: {e}")
    
    def test_submission_format(self, processed_data):
        """Test that submission files have correct format."""
        X_train, y, X_test = processed_data
        
        # Create a dummy submission
        submission = pd.DataFrame({
            'Id': range(1, len(X_test) + 1),
            'SalePrice': np.expm1(np.random.normal(12, 1, len(X_test)))
        })
        
        try:
            check_submission_format(submission, len(X_test))
        except Exception as e:
            pytest.fail(f"Submission format check failed: {e}")

