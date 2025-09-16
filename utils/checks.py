import pandas as pd
import numpy as np
from scipy import stats

#SalePrice and log1 Sale price sanity check
import numpy as np, pandas as pd

import config_local.local_config as cfg  

import numpy as np, pandas as pd
import config_local.local_config as cfg

def check():
    # --- Load
    train  = pd.read_csv(cfg.TRAIN_CSV)
    filled = pd.read_csv(cfg.TRAIN_FILLED_CSV)
    out_df = pd.read_csv(cfg.TRAIN_OUTLIER_FILLED_CSV)
    log_df = pd.read_csv(cfg.TRAIN_OUTLIER_FILLED_LOG1_CSV)              # has logSP only
    enc_df = pd.read_csv(cfg.TRAIN_OUTLIER_FILLED_LOG1_CATENCODED_CSV)   # has logSP + dummies

    print("📂 Files:")
    for n,p in [("train",cfg.TRAIN_CSV),("filled",cfg.TRAIN_FILLED_CSV),
                ("out",cfg.TRAIN_OUTLIER_FILLED_CSV),("log1",cfg.TRAIN_OUTLIER_FILLED_LOG1_CSV),
                ("encoded",cfg.TRAIN_OUTLIER_FILLED_LOG1_CATENCODED_CSV)]:
        print(f"  - {n:8s}: {p}")

    # --- Presence & dimensions
    for name, df, tgt in [
        ("train",   train,  "SalePrice"),
        ("filled",  filled, "SalePrice"),
        ("out",     out_df, "SalePrice"),
        ("log1",    log_df, "logSP"),        # <-- expect logSP here
        ("encoded", enc_df, "logSP"),        # <-- and here
    ]:
        if tgt not in df.columns:
            raise SystemExit(f"❌ {name} missing target col '{tgt}'")

    print("\nℹ️ Dimensions (rows | target non-null)")
    print(f"  train    | {len(train):5d} | {train['SalePrice'].notna().sum():5d}")
    print(f"  filled   | {len(filled):5d} | {filled['SalePrice'].notna().sum():5d}")
    print(f"  out      | {len(out_df):5d} | {out_df['SalePrice'].notna().sum():5d}")
    print(f"  log1     | {len(log_df):5d} | {log_df['logSP'].notna().sum():5d}")
    print(f"  encoded  | {len(enc_df):5d} | {enc_df['logSP'].notna().sum():5d}")

    # 1) train validity & parity with filled
    print("\n🔎 [1] train SalePrice validity & train↔filled parity")
    s = train["SalePrice"]
    assert s.notna().all() and (s > 0).all(), "❌ train.SalePrice must be positive with no NaNs"
    assert s.reset_index(drop=True).equals(filled["SalePrice"].reset_index(drop=True)), "❌ train ↔ filled mismatch"
    print("✅ train SalePrice valid; train ↔ filled identical")

    # 2) out is subset of train
    print("\n🔎 [2] outlier subset check (out ⊆ train by multiplicity)")
    vc_out, vc_train = out_df.SalePrice.value_counts(), train.SalePrice.value_counts()
    assert vc_out.le(vc_train.reindex(vc_out.index, fill_value=0)).all(), "❌ out has unknown/extra SalePrice values"
    print(f"✅ out ⊆ train ({len(train)-len(out_df)} rows removed)")

    # 3) log1 correctness: logSP == log1p(out.SalePrice)
    print("\n🔎 [3] log1: logSP vs log1p(out.SalePrice)")
    exp = np.log1p(out_df["SalePrice"].astype(float)).reset_index(drop=True)
    act = log_df["logSP"].astype(float).reset_index(drop=True)
    assert len(exp) == len(act), "❌ log1 row count differs from out"
    assert np.allclose(exp, act, rtol=1e-6, atol=1e-6), "❌ logSP ≠ log1p(out.SalePrice)"
    print("✅ logSP matches log1p(out.SalePrice)")

    # 4) encoded sanity (rows, numeric dtypes, target parity)
    print("\n🔎 [4] encoded sanity (rows, dtypes, target parity)")
    assert len(enc_df) == len(log_df), "❌ encoded row count mismatch vs log1"
    assert enc_df.select_dtypes(exclude="number").empty, "❌ encoded has non-numeric columns"
    assert np.allclose(enc_df["logSP"].to_numpy(dtype=float),
                       log_df["logSP"].to_numpy(dtype=float), rtol=1e-6, atol=1e-6), \
           "❌ encoded.logSP ≠ log1.logSP"
    print("✅ encoded rows & dtypes OK; logSP parity OK")

    # 5) numeric parity for shared numeric columns (excluding target)
    print("\n🔎 [5] numeric parity (log1 → encoded) for shared numeric cols (excl. logSP)")
    num_log = log_df.select_dtypes(include="number").columns.drop("logSP", errors="ignore")
    common_num = [c for c in num_log if c in enc_df.columns]
    bad = []
    for c in common_num:
        a, b = log_df[c].to_numpy(), enc_df[c].to_numpy()
        same = (np.array_equal(a,b) if np.issubdtype(a.dtype,np.integer) and np.issubdtype(b.dtype,np.integer)
                else np.allclose(a.astype(float), b.astype(float), rtol=1e-7, atol=1e-9))
        if not same: bad.append(c)
    assert not bad, f"❌ numeric columns changed during encoding: {bad[:6]}..."
    print(f"✅ numeric parity OK for {len(common_num)} columns")

    print("\n🎉 All checks passed")

DEFAULT_THRESHOLDS = {
    "abs_skew": 0.5,
    "abs_excess_kurt": 1.0,
    "jb_pvalue": 0.05,
    "ks_norm": 0.08,
}

def _ks_stat(s: pd.Series, dist_name: str) -> float:
    dist = getattr(stats, dist_name)
    params = dist.fit(s.dropna())
    ks_stat, _ = stats.kstest(s.dropna(), dist_name, args=params)
    return float(ks_stat)

def evaluate_many(series_dict: dict[str, pd.Series],
                  thresholds: dict[str, float] | None = None) -> pd.DataFrame:
    """Check distribution shape & fit for multiple series."""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    rows = []

    for name, s in series_dict.items():
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            continue

        # shape
        skew = stats.skew(s, bias=False)
        kurt_ex = stats.kurtosis(s, fisher=True, bias=False)
        jb_stat, jb_p = stats.jarque_bera(s)

        def add(metric, value, cmp_op="<=", note=""):
            thr = th.get(metric, np.nan)
            ok = (value <= thr) if cmp_op == "<=" else (value >= thr)
            rows.append([name, metric, float(value), float(thr), ok, note])

        add("abs_skew", abs(skew))
        add("abs_excess_kurt", abs(kurt_ex))
        rows.append([name, "jb_stat", float(jb_stat), np.nan, None, "smaller is better"])
        rows.append([name, "jb_pvalue", float(jb_p), th["jb_pvalue"], jb_p >= th["jb_pvalue"], ">= alpha passes"])
        add("ks_norm", _ks_stat(s, "norm"))

    out = pd.DataFrame(rows, columns=["series", "metric", "value", "threshold", "pass", "note"])
    out["value"] = out["value"].round(3)
    overall = (out.groupby("series")["pass"]
                 .apply(lambda c: c.dropna().all() if len(c.dropna()) else False)
                 .rename("overall_pass").reset_index())
    return out.merge(overall, on="series", how="left")

if __name__ == "__main__":
    check()