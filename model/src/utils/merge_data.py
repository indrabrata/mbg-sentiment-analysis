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

    out_file = output_dir / "mbg.csv"
    merged.to_csv(out_file, index=False, encoding="utf-8-sig")

    print("Merged files:")
    for f in selected:
        print(" -", f.name)

    print("Output saved:", out_file)
    return out_file
