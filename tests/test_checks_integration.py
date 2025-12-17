import os
import pytest

try:
    from config_local import local_config as cfg
    HAVE_CFG = True
except Exception:
    HAVE_CFG = False

@pytest.mark.integration
@pytest.mark.skipif(not HAVE_CFG, reason="config_local not available")
def test_check_runs():
    req = [
        getattr(cfg, "TRAIN_CSV", None),
        getattr(cfg, "TRAIN_FILLED_CSV", None),
        getattr(cfg, "TRAIN_OUTLIER_FILLED_CSV", None),
        getattr(cfg, "TRAIN_OUTLIER_FILLED_LOG1_CSV", None),
        getattr(cfg, "TRAIN_OUTLIER_FILLED_LOG1_CATENCODED_CSV", None),
    ]
    if not all(p and os.path.exists(p) for p in req):
        pytest.skip("dataset files not present")

    from utils.checks import check
    check()  # if something is wrong, it will raise