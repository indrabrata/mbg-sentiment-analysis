#!/usr/bin/env python3
import re, pandas as pd, argparse, logging, emoji
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def preprocess_tweet(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "HTTPURL", text)
    text = re.sub(r"@\w+", "@USER", text)
    text = emoji.demojize(text, language="en")
    text = re.sub(r"&\w+;", "", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_text(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    logging.info(f"Input rows = {len(df)}")
    if "lang" in df.columns:
        before = len(df)
        df = df[df["lang"].isin(["in", "id"])].copy()
        logging.info(f"Filter Bahasa Indonesia: {before} → {len(df)}")

    if "full_text" not in df.columns:
        raise KeyError("Kolom 'full_text' tidak ditemukan!")

    df["clean_text"] = df["full_text"].apply(preprocess_tweet)
    keep = [
        "full_text", "clean_text", "created_at", "username", "tweet_url",
        "retweet_count", "reply_count", "favorite_count",
        "location", "lang", "id_str", "source_file"
    ]
    exist = [c for c in keep if c in df.columns]
    out = df[exist].copy()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    logging.info(f"✅ Cleaned disimpan: {output_path} (rows={len(out)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Membersihkan teks tweet MBG.")
    parser.add_argument("--input", default="mbg-adjusted/data/processed/mbg_combined.csv")
    parser.add_argument("--output", default="mbg-adjusted/data/processed/mbg_cleaned.csv")
    args = parser.parse_args()
    clean_text(args.input, args.output)
