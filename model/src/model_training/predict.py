#!/usr/bin/env python3
from transformers import pipeline
import torch, argparse, logging, pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODEL_DIR = "results_indobertweet/best_model"
MODEL_NAME = "indolem/indobertweet-base-uncased"

def load_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("sentiment-analysis", model=MODEL_DIR, tokenizer=MODEL_NAME, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediksi sentimen MBG IndoBERTweet.")
    parser.add_argument("--input", help="Path CSV (opsional, kolom clean_text).")
    parser.add_argument("--output", default="data/predictions/pred_results.csv")
    args = parser.parse_args()

    model = load_model()
    Path("data/predictions").mkdir(parents=True, exist_ok=True)

    if args.input:
        df = pd.read_csv(args.input)
        preds = [model(t)[0] for t in df["clean_text"].tolist()]
        df["label"] = [p["label"] for p in preds]
        df["confidence"] = [p["score"] for p in preds]
        df.to_csv(args.output, index=False)
        logging.info(f"✅ Prediksi batch disimpan di {args.output}")
    else:
        texts = [
            "Program makan bergizi gratis sangat membantu siswa!",
            "Makanan yang dibagikan tidak layak konsumsi, sangat mengecewakan."
        ]
        for t in texts:
            res = model(t)[0]
            print(res)
