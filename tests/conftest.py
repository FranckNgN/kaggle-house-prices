"""Shared pytest fixtures for all tests."""
import pytest
import pandas as pd
from pathlib import Path

try:
    import config_local.local_config as cfg
    HAVE_CFG = True
except ImportError:
    HAVE_CFG = False
    cfg = None


@pytest.fixture(scope="session")
def config_available():
    """Check if config is available."""
    return HAVE_CFG


@pytest.fixture(scope="session")
def raw_data_available(config_available):
    """Check if raw data files exist."""
    if not config_available:
        return False
    return (
        Path(cfg.TRAIN_CSV).exists() and
        Path(cfg.TEST_CSV).exists()
    )


@pytest.fixture(scope="session")
def processed_data_available(config_available):
    """Check if processed data (stage 6) exists."""
    if not config_available:
        return False
    return (
        Path(cfg.TRAIN_PROCESS6_CSV).exists() and
        Path(cfg.TEST_PROCESS6_CSV).exists()
    )


@pytest.fixture
def load_stage_data():
    """Fixture to load data for a specific stage."""
    def _load(stage_num: int):
        if not HAVE_CFG:
            pytest.skip("config_local not available")
        
        train_attr = f"TRAIN_PROCESS{stage_num}_CSV"
        test_attr = f"TEST_PROCESS{stage_num}_CSV"
        
        train_path = getattr(cfg, train_attr, None)
        test_path = getattr(cfg, test_attr, None)
        
        if train_path is None or test_path is None:
            pytest.skip(f"Stage {stage_num} paths not found in config")
        
        if not Path(train_path).exists() or not Path(test_path).exists():
            pytest.skip(f"Stage {stage_num} output files not found")
        
        return pd.read_csv(train_path), pd.read_csv(test_path)
    
    return _load

