#!/usr/bin/env python3
"""
trainer.py
Incremental fine-tuning IndoBERTweet with expert-labelled weekly data
+ partial layer freezing
+ evaluation (accuracy, F1-macro)
+ MLflow metric logging
"""

import os
import logging
from pathlib import Path

import pandas as pd
from datasets import Dataset

import mlflow
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from .freezing import freeze_lower_layers
from .metrics import compute_metrics


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_MODEL = "indolem/indobertweet-base-uncased"


def load_dataset(csv_path: str) -> Dataset:
    df = pd.read_csv(csv_path)

    required_cols = {"text", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns {required_cols}")

    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    return Dataset.from_pandas(df, preserve_index=False)


def tokenize_dataset(ds: Dataset, tokenizer, max_length: int = 128):
    return ds.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        ),
        batched=True
    )


def train(args):
    logging.info("🚀 Incremental training started")

    # Training configuration
    max_length = 128
    learning_rate = 2e-5
    batch_size = 16
    weight_decay = 0.01
    num_labels = 3


    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))

    with mlflow.start_run(run_name="incremental-indobertweet"):

        # Log experiment config
        mlflow.log_params({
            "base_model": BASE_MODEL,
            "epochs": args.epochs,
            "learning_rate": learning_rate,
            "freeze_layers": args.freeze_layers,
            "batch_size": batch_size,
            "max_length": max_length,
            "weight_decay": weight_decay
        })


        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        logging.info("📦 Loading previous model from: %s", args.model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_dir,
            num_labels=num_labels
        )

        if args.freeze_layers > 0:
            freeze_lower_layers(model, args.freeze_layers)

        train_ds = tokenize_dataset(load_dataset(args.train), tokenizer, max_length)
        val_ds   = tokenize_dataset(load_dataset(args.val), tokenizer, max_length)

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=True,

            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,

            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,

            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,

            logging_steps=50,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )


        trainer.train()

        metrics = trainer.evaluate()
        logging.info("📊 Evaluation metrics: %s", metrics)

        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        logging.info("💾 Model saved to %s", output_dir)
        logging.info("✅ Incremental training finished successfully")
