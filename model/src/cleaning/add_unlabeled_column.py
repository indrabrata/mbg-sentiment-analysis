#!/usr/bin/env python3
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add empty label column for expert annotator.")
    parser.add_argument("--input", required=True, help="Path to cleaned CSV")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--date", required=False)

    args = parser.parse_args()
    date_str = args.date or datetime.now().strftime("%Y-%m-%d")

    add_label_column(args.input, args.output_dir, date_str)
