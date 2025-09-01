import pandas as pd
import numpy as np
from config_local import local_config as cfg

def check():
    # --- Load datasets using config paths ---
    train       = pd.read_csv(cfg.TRAIN_CSV)
    train_fill  = pd.read_csv(cfg.TRAIN_FILLED_CSV)
    train_out   = pd.read_csv(cfg.TRAIN_OUTLIER_FILLED_CSV)
    train_out_log = pd.read_csv(cfg.TRAIN_OUTLIER_FILLED_LOG1)

    # --- Helper for dimensions ---
    def dims(df): 
        return len(df), df["SalePrice"].notna().sum()

    # --- Dimension + presence check ---
    for name, df in [
        ("train", train),
        ("train_filled", train_fill),
        ("train_outlier_filled", train_out),
        ("train_outlier_filled_log1", train_out_log)
    ]:
        if "SalePrice" not in df.columns:
            raise SystemExit(f"❌ {name} missing 'SalePrice' column")

    print("ℹ️ Rows / SalePrice non-null:")
    for name, df in [
        ("train", train),
        ("train_filled", train_fill),
        ("train_outlier_filled", train_out),
        ("train_outlier_filled_log1", train_out_log)
    ]:
        r, nn = dims(df)
        print(f"  {name:26s}: {r} / {nn}")

    # --- Validity on base train ---
    if train["SalePrice"].isnull().any():
        raise SystemExit("❌ NaNs found in SalePrice (train)")
    if (train["SalePrice"] <= 0).any():
        raise SystemExit("❌ Non-positive values found in SalePrice (train)")

    # --- Consistency: train vs train_filled ---
    if not train["SalePrice"].reset_index(drop=True).equals(
        train_fill["SalePrice"].reset_index(drop=True)
    ):
        raise SystemExit("❌ SalePrice mismatch between TRAIN_CSV and TRAIN_FILLED_CSV")

    # --- Outlier file must be a subset of train ---
    if len(train_out) > len(train):
        raise SystemExit("❌ train_outlier_filled has more rows than train (should be ≤)")
    vc_out   = train_out["SalePrice"].value_counts()
    vc_train = train["SalePrice"].value_counts()
    if not vc_out.le(vc_train.reindex(vc_out.index, fill_value=0)).all():
        raise SystemExit("❌ train_outlier_filled contains SalePrice values not present (or too many) vs train")
    print(f"✅ Outlier subset check passed ({len(train) - len(train_out)} rows removed)")

    # --- Log1p check ---
    expected = np.log1p(train_out["SalePrice"].reset_index(drop=True).astype(float))
    actual   = train_out_log["SalePrice"].reset_index(drop=True).astype(float)

    if len(expected) != len(actual):
        raise SystemExit("❌ train_outlier_filled_log1 length differs from train_outlier_filled")
    if not np.allclose(expected.values, actual.values, rtol=1e-6, atol=1e-6):
        raise SystemExit("❌ train_outlier_filled_log1 is not log1p of train_outlier_filled")
    print("✅ Log1p check passed (train_outlier_filled_log1 matches log transform)")

    print("✅ All checks passed")

if __name__ == "__main__":
    check()