import re, pandas as pd, argparse, logging, emoji
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

    # Filter bahasa Indonesia tetapi tidak menghapus kolom lain
    if "lang" in df.columns:
        before = len(df)
        df = df[df["lang"].isin(["in", "id"])].copy()
        logging.info(f"Filter bahasa Indonesia: {before} → {len(df)}")

    # Check kolom full_text
    if "full_text" not in df.columns:
        raise KeyError("Kolom 'full_text' tidak ditemukan!")

    # Tambahkan clean_text TANPA menghapus kolom lain
    df["clean_text"] = df["full_text"].apply(preprocess_tweet)

    # Simpan semua kolom original + clean_text
    output_path = Path(output_dir) / "mbg_cleaned.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logging.info(f"✅ Cleaned saved to: {output_path} (rows={len(df)})")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleaning text MBG (Scraped Twitter).")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    clean_text(args.input, args.output_dir)
