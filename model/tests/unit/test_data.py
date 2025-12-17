import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/weekly/mbg.csv")

def test_dataset_file_exists():
    assert DATA_PATH.exists(), "Dataset mbg.csv tidak ditemukan"

def test_required_columns_exist():
    df = pd.read_csv(DATA_PATH)
    assert {"clean_text", "label"}.issubset(df.columns)

def test_no_null_values():
    df = pd.read_csv(DATA_PATH)
    assert df["clean_text"].notna().all()
    assert df["label"].notna().all()

def test_label_value_range():
    df = pd.read_csv(DATA_PATH)
    assert df["label"].isin([0, 1, 2]).all()

def test_minimum_dataset_size():
    df = pd.read_csv(DATA_PATH)
    assert len(df) >= 100, "Dataset terlalu kecil untuk training"
