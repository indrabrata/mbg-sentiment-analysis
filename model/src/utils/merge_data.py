from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd


def merge_weekly(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now().date()
    week_start = today - timedelta(days=7)

    selected = []

    for f in input_dir.glob("mbg_labeled_*.csv"):
        try:
            date_str = f.stem.replace("mbg_labeled_", "")
            file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception as e:
            print(f"Skipping file {f}: {e}")
            continue

        if week_start <= file_date <= today:
            selected.append(f)

    if not selected:
        print("No labelled files in the last 7 days.")
        return None

    dfs = [pd.read_csv(f) for f in selected]
    merged = pd.concat(dfs, ignore_index=True)

    required_cols = {"clean_text", "label"}
    if not required_cols.issubset(merged.columns):
        raise ValueError(f"Dataset harus mengandung kolom {required_cols}")

    before = len(merged)
    merged = merged.dropna(subset=["label"])
    after_dropna = len(merged)

    merged["label"] = merged["label"].astype(int)

    merged = merged[merged["label"].isin([0, 1, 2])]
    after_range = len(merged)

    print(f"Rows before cleaning : {before}")
    print(f"After dropna(label)  : {after_dropna}")
    print(f"After label range   : {after_range}")

    if merged.empty:
        raise ValueError("❌ Semua data terfilter — tidak ada data valid untuk training")

    out_file = output_dir / "mbg.csv"
    merged.to_csv(out_file, index=False, encoding="utf-8-sig")

    print("Merged files:")
    for f in selected:
        print(" -", f.name)

    print("✅ Weekly dataset saved:", out_file)
    return out_file
