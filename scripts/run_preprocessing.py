import os
import papermill as pm

src_dir = "notebooks/preprocessing"
out_dir = "runs/latest"
os.makedirs(out_dir, exist_ok=True)

files = [
    "1cleaning.ipynb",
    "2outlier.ipynb",
    "3skewKurtosis.ipynb",
    "4categorialEncode.ipynb"
]

for f in files:
    in_path = os.path.join(src_dir, f)
    out_path = os.path.join(out_dir, f.replace(".ipynb", "_out.ipynb"))
    print(f"▶ Running {in_path}")
    pm.execute_notebook(in_path, out_path)
    print(f"✅ Wrote {out_path}")