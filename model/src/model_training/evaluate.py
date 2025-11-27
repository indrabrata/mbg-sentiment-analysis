#!/usr/bin/env python3
# evaluate_and_export.py
import pandas as pd, numpy as np, json, logging, os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def evaluate_model(test_path, model_dir, outdir):
    df = pd.read_csv(test_path)
    ds = Dataset.from_pandas(df)
    tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
    ds = ds.map(lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=128), batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    trainer = Trainer(model=model)
    preds = trainer.predict(ds)
    pred_labels = np.argmax(preds.predictions, axis=1)

    acc = accuracy_score(df["label"], pred_labels)
    p, r, f1, _ = precision_recall_fscore_support(df["label"], pred_labels, average="weighted", zero_division=0)
    metrics = {"accuracy": float(acc), "precision": float(p), "recall": float(r), "f1_score": float(f1)}

    Path(outdir).mkdir(parents=True, exist_ok=True)
    # json metrics
    with open(Path(outdir) / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # classification report (per class)
    report_dict = classification_report(df["label"], pred_labels, output_dict=True, zero_division=0)
    # convert to dataframe and save CSV
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(Path(outdir) / "classification_report.csv", index=True)

    # confusion matrix (csv + png)
    cm = confusion_matrix(df["label"], pred_labels)
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(Path(outdir) / "confusion_matrix.csv", index=False)

    # plot heatmap and save png
    try:
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(Path(outdir) / "confusion_matrix.png")
        plt.close()
    except Exception:
        # fallback: simple matplotlib
        plt.imshow(cm, interpolation='nearest')
        plt.colorbar()
        plt.savefig(Path(outdir) / "confusion_matrix.png")
        plt.close()

    logging.info(f"✅ Evaluasi selesai. Metrics: {metrics}, files in: {outdir}")

if __name__ == "__main__":
    evaluate_model("data/processed/test.csv", "results_indobertweet/best_model", "results_indobertweet/eval")
