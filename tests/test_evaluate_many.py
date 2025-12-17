import numpy as np
import pandas as pd
from utils.checks import evaluate_many

def test_evaluate_many_basic():
    series = {
        "normal": pd.Series(np.random.default_rng(42).normal(size=500)),
        "skewed": pd.Series(np.random.default_rng(1).exponential(size=500)),
    }
    df = evaluate_many(series)
    assert not df.empty
    assert "overall_pass" in df.columns