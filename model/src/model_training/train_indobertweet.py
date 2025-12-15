#!/usr/bin/env python3
"""
train_indobertweet.py
Incremental fine-tuning IndoBERTweet with expert-labelled weekly data
+ partial layer freezing
+ evaluation (accuracy, F1-macro)
+ MLflow metric logging
"""

import os
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

import mlflow
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# ----------------------------------------------------
# LOGGING
# ----------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_MODEL = "indolem/indobertweet-base-uncased"


# ----------------------------------------------------
# DATA LOADING
# ----------------------------------------------------
def load_dataset(csv_path: str) -> Dataset:
    df = pd.read_csv(csv_path)

    required_cols = {"text", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns {required_cols}")

    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    return Dataset.from_pandas(df, preserve_index=False)


# ----------------------------------------------------
# TOKENIZATION
# ----------------------------------------------------
def tokenize_dataset(ds: Dataset, tokenizer):
    return ds.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        ),
        batched=True
    )


# ----------------------------------------------------
# FREEZE LOWER LAYERS (CONTINUAL LEARNING CORE)
# ----------------------------------------------------
def freeze_lower_layers(model, freeze_until: int):
    """
    Freeze encoder layers [0 .. freeze_until-1]
    """
    for layer in model.bert.encoder.layer[:freeze_until]:
        for param in layer.parameters():
            param.requires_grad = False

    logging.info("🔒 Frozen bottom %d encoder layers", freeze_until)


# ----------------------------------------------------
# METRICS
# ----------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }


# ----------------------------------------------------
# TRAINING PIPELINE
# ----------------------------------------------------
def train(args):
    logging.info("🚀 Incremental training started")

    # -------------------------
    # MLflow setup (CLIENT)
    # -------------------------
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    with mlflow.start_run(run_name="incremental-indobertweet"):

        # Log experiment config
        mlflow.log_params({
            "base_model": BASE_MODEL,
            "epochs": args.epochs,
            "learning_rate": 2e-5,
            "freeze_layers": args.freeze_layers,
            "batch_size": 16
        })

        # -------------------------
        # Load tokenizer & model
        # -------------------------
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        logging.info("📦 Loading previous model from: %s", args.model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_dir,
            num_labels=3
        )

        if args.freeze_layers > 0:
            freeze_lower_layers(model, args.freeze_layers)

        # -------------------------
        # Load & tokenize datasets
        # -------------------------
        train_ds = tokenize_dataset(load_dataset(args.train), tokenizer)
        val_ds   = tokenize_dataset(load_dataset(args.val), tokenizer)

        # -------------------------
        # Training arguments
        # -------------------------
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=True,

            learning_rate=2e-5,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,

            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,

            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,

            logging_steps=50,
            report_to="none"
        )

        # -------------------------
        # Trainer
        # -------------------------
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        # -------------------------
        # Train
        # -------------------------
        trainer.train()

        # -------------------------
        # Final evaluation & MLflow logging
        # -------------------------
        metrics = trainer.evaluate()
        logging.info("📊 Evaluation metrics: %s", metrics)

        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

        # -------------------------
        # Save model
        # -------------------------
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        logging.info("💾 Model saved to %s", output_dir)
        logging.info("✅ Incremental training finished successfully")


# ----------------------------------------------------
# CLI
# ----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Training CSV (weekly expert-labelled)")
    parser.add_argument("--val", required=True, help="Validation CSV")
    parser.add_argument("--model_dir", required=True, help="Previous model directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for new model")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--freeze_layers", type=int, default=6)

    args = parser.parse_args()
    train(args)
