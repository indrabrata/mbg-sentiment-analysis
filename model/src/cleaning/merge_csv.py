#!/usr/bin/env python3
import pandas as pd
import glob, os, argparse, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def merge_all_csv(input_folder: str, output_path: str):
    files = glob.glob(os.path.join(input_folder, "*.csv"))
    if not files:
        raise FileNotFoundError(f"Tidak ada file .csv di: {input_folder}")

    logging.info(f"Ditemukan {len(files)} file CSV.")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            logging.warning(f"Gagal membaca {f}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    if "id_str" in combined.columns:
        before = len(combined)
        combined.drop_duplicates(subset="id_str", inplace=True)
        logging.info(f"Hapus duplikat id_str: {before} → {len(combined)}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")
    logging.info(f"✅ Gabungan disimpan: {output_path} (rows={len(combined)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gabungkan seluruh file CSV mentah.")
    parser.add_argument("--input", default="mbg-adjusted/data/raw", help="Folder input CSV mentah.")
    parser.add_argument("--output", default="mbg-adjusted/data/processed/mbg_combined.csv", help="Path output hasil gabungan.")
    args = parser.parse_args()
    merge_all_csv(args.input, args.output)
