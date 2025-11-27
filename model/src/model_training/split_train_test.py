#!/usr/bin/env python3
import pandas as pd, argparse, logging
from sklearn.model_selection import train_test_split
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def split_dataset(input_path: str):
    # --- Load data ---
    df = pd.read_csv(input_path).dropna(subset=["clean_text","label"])
    df = df[df["clean_text"].str.strip() != ""]
    total_rows = len(df)
    logging.info(f"Total data awal: {total_rows}")

    # --- Split Train / Val / Test ---
    X_train, X_temp, y_train, y_temp = train_test_split(
        df["clean_text"], df["label"],
        test_size=0.3, stratify=df["label"], random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5, stratify=y_temp, random_state=42
    )

    # --- Save hasil ---
    Path("data/processed").mkdir(exist_ok=True)
    train_path = "data/processed/train.csv"
    val_path   = "data/processed/val.csv"
    test_path  = "data/processed/test.csv"

    pd.DataFrame({"text": X_train, "label": y_train}).to_csv(train_path, index=False)
    pd.DataFrame({"text": X_val, "label": y_val}).to_csv(val_path, index=False)
    pd.DataFrame({"text": X_test, "label": y_test}).to_csv(test_path, index=False)

    # --- Logging jumlah hasil split ---
    logging.info("Dataset split selesai!")
    logging.info(f"Train set : {len(X_train)} rows ({len(X_train)/total_rows:.1%})")
    logging.info(f"Val set   : {len(X_val)} rows ({len(X_val)/total_rows:.1%})")
    logging.info(f"Test set  : {len(X_test)} rows ({len(X_test)/total_rows:.1%})")

    return train_path, val_path, test_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset menjadi Train/Val/Test.")
    parser.add_argument("--input", default="mbg-adjusted/data/processed/mbg_prepared.csv")
    args = parser.parse_args()
    split_dataset(args.input)
