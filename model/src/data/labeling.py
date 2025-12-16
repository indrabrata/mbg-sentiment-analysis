import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def add_label_column(cleaned_path: str, output_dir: str, date_str: str):
    df = pd.read_csv(cleaned_path)

    df["label"] = ""

    out_path = Path(output_dir) / f"mbg_cleaned_unlabeled_{date_str}.csv"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    logging.info(f"📄 Unlabeled file created: {out_path} (rows={len(df)})")

    return out_path
