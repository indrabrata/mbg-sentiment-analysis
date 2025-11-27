#!/usr/bin/env python3
import pandas as pd, torch, argparse, logging
from transformers import pipeline
from tqdm import tqdm
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODEL_NAME = "w11wo/indonesian-roberta-base-sentiment-classifier"
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}

def auto_label(input_path: str, output_path: str, batch_size: int = 64):
    df = pd.read_csv(input_path)
    device = 0 if torch.cuda.is_available() else -1
    logging.info(f"Menjalankan model {MODEL_NAME} di: {'GPU' if device == 0 else 'CPU'}")
    clf = pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME, device=device)
    texts = df["clean_text"].fillna("").astype(str).tolist()
    predictions = []

    for i in tqdm(range(0, len(texts), batch_size)):
        try:
            results = clf(texts[i:i+batch_size])
            predictions.extend([LABEL_MAP.get(r["label"], 1) for r in results])
        except Exception as e:
            logging.warning(f"Error pada batch {i//batch_size}: {e}")
            predictions.extend([1]*len(texts[i:i+batch_size]))

    df["label"] = predictions
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logging.info(f"✅ Auto-labeling selesai → {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-label dataset MBG dengan model RoBERTa.")
    parser.add_argument("--input", default="mbg-adjusted/data/processed/mbg_cleaned.csv")
    parser.add_argument("--output", default="mbg-adjusted/data/processed/mbg_prepared.csv")
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    auto_label(args.input, args.output, args.batch)
