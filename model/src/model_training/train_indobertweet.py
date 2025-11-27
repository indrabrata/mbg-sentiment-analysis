#!/usr/bin/env python3
# train_and_export.py
import os, pandas as pd, torch, argparse, logging, joblib, json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, pipeline
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def train_model(train_path, val_path, outdir, model_name="indolem/indobertweet-base-uncased"):
    train_df, val_df = pd.read_csv(train_path), pd.read_csv(val_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds, val_ds = Dataset.from_pandas(train_df), Dataset.from_pandas(val_df)
    train_ds = train_ds.map(lambda e: tokenizer(e["text"], truncation=True, max_length=128), batched=True)
    val_ds   = val_ds.map(lambda e: tokenizer(e["text"], truncation=True, max_length=128), batched=True)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    args = TrainingArguments(
        output_dir=outdir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        num_train_epochs=4,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=128,
        warmup_ratio=0.1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none"
    )

    def compute_metrics(pred):
        labels, preds = pred.label_ids, pred.predictions.argmax(-1)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics
    )
    trainer.train()

    # Simpan HF model + tokenizer (dir usable untuk load_pretrained)
    best_dir = Path(outdir) / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(best_dir.as_posix())
    tokenizer.save_pretrained(best_dir.as_posix())

    # Prediksi val untuk metrics final
    preds = trainer.predict(val_ds)
    pred_labels = preds.predictions.argmax(-1)
    f1_val = f1_score(preds.label_ids, pred_labels, average="weighted")
    acc_val = accuracy_score(preds.label_ids, pred_labels)

    # Buat pipeline inference dan dump dengan joblib agar mudah dipakai di FastAPI
    device = 0 if torch.cuda.is_available() else -1
    inf_pipeline = pipeline("sentiment-analysis", model=best_dir.as_posix(), tokenizer=best_dir.as_posix(), device=device)
    joblib_path = Path(outdir) / "model_pipeline.joblib"
    joblib.dump(inf_pipeline, joblib_path.as_posix(), compress=3)

    # Simpan metrics training ke file
    metrics = {"val_f1": float(f1_val), "val_accuracy": float(acc_val)}
    with open(Path(outdir) / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logging.info(f"✅ Training selesai. model dir: {best_dir}, joblib: {joblib_path}, metrics: {metrics}")
    return best_dir.as_posix(), joblib_path.as_posix()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune IndoBERTweet and export joblib pipeline.")
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--val", default="data/processed/val.csv")
    parser.add_argument("--outdir", default="results_indobertweet")
    args = parser.parse_args()
    train_model(args.train, args.val, args.outdir)
