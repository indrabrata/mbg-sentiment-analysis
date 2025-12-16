import re
import pandas as pd
import logging
import emoji
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def preprocess_tweet(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "HTTPURL", text)
    text = re.sub(r"@\w+", "@USER", text)
    text = emoji.demojize(text, language="en")
    text = re.sub(r"&\w+;", "", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text(input_path: str, output_dir: str):
    df = pd.read_csv(input_path)
    logging.info(f"Input rows = {len(df)}")

    if "lang" in df.columns:
        before = len(df)
        df = df[df["lang"].isin(["in", "id"])].copy()
        logging.info(f"Filter bahasa Indonesia: {before} → {len(df)}")

    if "full_text" not in df.columns:
        raise KeyError("Kolom 'full_text' tidak ditemukan!")

    df["clean_text"] = df["full_text"].apply(preprocess_tweet)

    output_path = Path(output_dir) / "mbg_cleaned.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logging.info(f"✅ Cleaned saved to: {output_path} (rows={len(df)})")

    return output_path
