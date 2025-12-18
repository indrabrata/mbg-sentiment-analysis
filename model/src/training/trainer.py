#!/usr/bin/env python3
"""
trainer.py
Incremental fine-tuning IndoBERTweet with expert-labelled weekly data
+ partial layer freezing
+ evaluation (accuracy, F1-macro)
+ MLflow metric logging
+ model performance gate BEFORE MLflow model logging
"""

import os
import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset

import mlflow
import mlflow.transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline
)

from .freezing import freeze_lower_layers
from .metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MIN_F1_MACRO = float(os.getenv("MIN_F1_MACRO", "0.80"))

def load_dataset(csv_path: str) -> Dataset:
    df = pd.read_csv(csv_path)

    required_cols = {"clean_text", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns {required_cols}")

    df = df.dropna(subset=["clean_text", "label"])
    df["label"] = df["label"].astype(int)

    return Dataset.from_pandas(df, preserve_index=False)


def tokenize_dataset(ds: Dataset, tokenizer, max_length: int = 128):
    return ds.map(
        lambda x: tokenizer(
            x["clean_text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        ),
        batched=True
    )

def check_previous_model_exists(experiment_name: str) -> tuple:
    """
    Check if there's a previous model in MLflow experiment.
    Returns: (bool: exists, str: model_uri or None)
    """
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)

        if not experiment:
            return False, None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )

        if not runs:
            return False, None

        run_id = runs[0].info.run_id
        artifacts = client.list_artifacts(run_id, path="model")

        if not artifacts:
            return False, None

        return True, f"runs:/{run_id}/model"

    except Exception as e:
        logging.warning(f"⚠️ Error checking previous model: {e}")
        return False, None


def train(args):
    logging.info("🚀 Training started")

    max_length = int(os.getenv("MAX_LENGTH", "128"))
    learning_rate = float(os.getenv("LEARNING_RATE", "2e-5"))
    batch_size = int(os.getenv("BATCH_SIZE", "16"))
    weight_decay = float(os.getenv("WEIGHT_DECAY", "0.01"))
    epochs = int(os.getenv("EPOCHS", "3"))
    freeze_layers = int(os.getenv("FREEZE_LAYERS", "6"))
    validation_split = float(os.getenv("VALIDATION_SPLIT", "0.2"))
    num_labels = int(os.getenv("NUM_LABELS", "3"))

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "mbg-sentiment-analysis")
    base_model = os.getenv("BASE_MODEL", "indolem/indobertweet-base-uncased")
    model_name = os.getenv("MODEL_NAME", "mbg-sentiment-analysis-model")

    mlflow.set_experiment(experiment_name)

    has_previous_model, previous_model_uri = check_previous_model_exists(experiment_name)

    if has_previous_model:
        import tempfile
        temp_dir = tempfile.mkdtemp()
        model_source = mlflow.artifacts.download_artifacts(
            artifact_uri=previous_model_uri,
            dst_path=temp_dir
        )
        is_incremental = True
        logging.info(f"📥 Using previous model for incremental training")
    else:
        model_source = base_model
        is_incremental = False
        logging.info(f"🆕 Starting fresh training from base model")

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_type = "incremental" if is_incremental else "fresh"
    output_dir = Path(f"models/{training_type}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"📁 Output directory: {output_dir}")

    with mlflow.start_run(run_name="mbg-sentiment-analysis-train"):

        mlflow.log_params({
            "base_model": base_model,
            "is_incremental": is_incremental,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "freeze_layers": freeze_layers if is_incremental else 0,
            "batch_size": batch_size,
            "max_length": max_length,
            "weight_decay": weight_decay,
            "validation_split": validation_split,
            "min_f1_macro_threshold": MIN_F1_MACRO
        })

        logging.info(f"📂 Loading training data from: {args.train_data}")
        full_dataset = load_dataset(args.train_data)

        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(
            range(len(full_dataset)),
            test_size=validation_split,
            random_state=42,
            stratify=full_dataset["label"]
        )

        train_ds_raw = full_dataset.select(train_idx)
        val_ds_raw = full_dataset.select(val_idx)
        logging.info(f"📊 Train size: {len(train_ds_raw)}, Validation size: {len(val_ds_raw)}")

        tokenizer = AutoTokenizer.from_pretrained(base_model)

        train_ds = tokenize_dataset(train_ds_raw, tokenizer, max_length)
        val_ds = tokenize_dataset(val_ds_raw, tokenizer, max_length)

        logging.info(f"📦 Loading model from: {model_source}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            num_labels=num_labels
        )

        if is_incremental and freeze_layers > 0:
            freeze_lower_layers(model, freeze_layers)
            logging.info(f"🔒 Incremental training mode: {freeze_layers} layers frozen")
        else:
            logging.info(f"🆕 Fresh training mode: all layers trainable")

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            overwrite_output_dir=True,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy="epoch",
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

        logging.info("🏋️ Starting training...")
        trainer.train()

        logging.info("📊 Evaluating model...")
        metrics = trainer.evaluate()
        logging.info("📊 Evaluation metrics: %s", metrics)

        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

        # -------------------------
        # Model Performance Gate
        # -------------------------
        eval_f1 = float(metrics.get("eval_f1_macro", 0.0))

        if eval_f1 < MIN_F1_MACRO:
            mlflow.log_param("model_accepted", False)
            mlflow.set_tag("model_status", "rejected")
            logging.error(
                "❌ MODEL REJECTED: eval_f1_macro=%.4f < threshold=%.2f",
                eval_f1,
                MIN_F1_MACRO
            )
            logging.error("   Model will NOT be logged to MLflow")
            return

        mlflow.log_param("model_accepted", True)
        mlflow.set_tag("model_status", "accepted")
        logging.info(
            "✅ MODEL ACCEPTED: eval_f1_macro=%.4f >= threshold=%.2f",
            eval_f1,
            MIN_F1_MACRO
        )

        # -------------------------
        # Save model locally
        # -------------------------
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logging.info(f"💾 Model saved locally to {output_dir}")

        # -------------------------
        # Create pipeline and log to MLflow
        # -------------------------
        try:
            logging.info("📦 Creating sentiment analysis pipeline...")
            sentiment_pipeline = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=-1  # CPU
            )

            logging.info("📤 Logging model to MLflow...")
            mlflow.transformers.log_model(
                transformers_model=sentiment_pipeline,
                artifact_path="model",
                registered_model_name=model_name
            )

            logging.info("✅ Model logged to MLflow and registered as: %s", model_name)

        except Exception as e:
            logging.error(f"❌ Failed to log model to MLflow: {e}")
            import traceback
            logging.error(traceback.format_exc())
            raise

        logging.info("✅ Training completed successfully!")
