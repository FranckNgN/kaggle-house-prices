"""Tests to validate all generated submission files."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from utils.checks import check_submission_format
from utils.model_validation import validate_submission

try:
    import config_local.local_config as cfg
    HAVE_CFG = True
except ImportError:
    HAVE_CFG = False


@pytest.mark.integration
@pytest.mark.skipif(not HAVE_CFG, reason="config_local not available")
class TestSubmissionFiles:
    """Tests to validate all submission files."""
    
    @pytest.fixture
    def sample_submission(self):
        """Load sample submission to get expected IDs and size."""
        if not Path(cfg.SAMPLE_SUBMISSION_CSV).exists():
            pytest.skip("Sample submission not found")
        return pd.read_csv(cfg.SAMPLE_SUBMISSION_CSV)
    
    def test_sample_submission_format(self, sample_submission):
        """Test that sample submission has correct format."""
        assert 'Id' in sample_submission.columns, "Sample submission missing 'Id' column"
        assert 'SalePrice' in sample_submission.columns, "Sample submission missing 'SalePrice' column"
        assert len(sample_submission) > 0, "Sample submission is empty"
    
    def test_all_submission_files_valid(self, sample_submission):
        """Test that all submission files in data/submissions are valid."""
        submissions_dir = Path(cfg.SUBMISSIONS_DIR)
        if not submissions_dir.exists():
            pytest.skip("Submissions directory not found")
        
        # Find all CSV files except sample_submission
        submission_files = [
            f for f in submissions_dir.rglob("*.csv")
            if f.name != "sample_submission.csv"
        ]
        
        if not submission_files:
            pytest.skip("No submission files found")
        
        expected_size = len(sample_submission)
        expected_ids = sample_submission['Id'].values
        
        failed_submissions = []
        
        for sub_file in submission_files:
            try:
                submission = pd.read_csv(sub_file)
                
                # Check format
                check_submission_format(submission, expected_size)
                
                # Check ID matching
                if 'Id' in submission.columns:
                    submission_ids = submission['Id'].values
                    if not np.array_equal(submission_ids, expected_ids):
                        # Check if it's just order or actual mismatch
                        if set(submission_ids) != set(expected_ids):
                            failed_submissions.append({
                                'file': sub_file.name,
                                'error': 'ID mismatch - missing or extra IDs'
                            })
                        else:
                            failed_submissions.append({
                                'file': sub_file.name,
                                'error': 'ID order mismatch'
                            })
                
                # Check prediction values
                if 'SalePrice' in submission.columns:
                    prices = submission['SalePrice'].values
                    if not np.isfinite(prices).all():
                        failed_submissions.append({
                            'file': sub_file.name,
                            'error': 'Non-finite values in SalePrice'
                        })
                    elif (prices <= 0).any():
                        failed_submissions.append({
                            'file': sub_file.name,
                            'error': 'Non-positive values in SalePrice'
                        })
                    elif np.std(prices) < 1e-6:
                        failed_submissions.append({
                            'file': sub_file.name,
                            'error': 'All predictions are identical'
                        })
                
            except Exception as e:
                failed_submissions.append({
                    'file': sub_file.name,
                    'error': str(e)
                })
        
        if failed_submissions:
            error_msg = "Failed submissions:\n"
            for fail in failed_submissions:
                error_msg += f"  - {fail['file']}: {fail['error']}\n"
            pytest.fail(error_msg)
        
        # If we get here, all submissions passed
        assert len(submission_files) > 0, "No submission files to validate"

