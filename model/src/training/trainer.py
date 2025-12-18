#!/usr/bin/env python3
"""
trainer.py
Incremental fine-tuning IndoBERTweet with expert-labelled weekly data
+ partial layer freezing
+ evaluation (accuracy, F1-macro)
+ MLflow metric logging
+ model performance gate
"""

import os
import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset

import mlflow
import mlflow.pytorch
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

        # Get experiment
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            logging.info("📭 No previous experiment found. Starting fresh training.")
            return False, None

        # Search for runs with logged models
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="",
            order_by=["start_time DESC"],
            max_results=1
        )

        if not runs:
            logging.info("📭 No previous runs found. Starting fresh training.")
            return False, None

        latest_run = runs[0]
        run_id = latest_run.info.run_id

        # Check if this run has model artifacts
        artifacts = client.list_artifacts(run_id, path="model")
        if not artifacts:
            logging.info("📭 No model artifacts in latest run. Starting fresh training.")
            return False, None

        # Model exists, return the URI
        model_uri = f"runs:/{run_id}/model"
        logging.info(f"✅ Found previous model: {model_uri}")
        return True, model_uri

    except Exception as e:
        logging.warning(f"⚠️ Error checking for previous model: {e}")
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
    min_f1_macro = float(os.getenv("MIN_F1_MACRO", "0.70"))


    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "mbg-sentiment-analysis")

    # Set or create experiment
    try:
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logging.warning(f"Could not set experiment: {e}")

 
    has_previous_model, previous_model_uri = check_previous_model_exists(experiment_name)

    if has_previous_model:
        # Download previous model from MLflow
        import tempfile
        temp_dir = tempfile.mkdtemp()
        model_source = mlflow.artifacts.download_artifacts(
            artifact_uri=previous_model_uri,
            dst_path=temp_dir
        )
        is_incremental = True
        logging.info(f"📥 Downloaded previous model from MLflow for incremental training")
    else:
        # No previous model, train from scratch
        model_source = BASE_MODEL
        is_incremental = False
        logging.info(f"🆕 Starting fresh training from base model: {BASE_MODEL}")

    # -------------------------
    # Auto-generate output directory with timestamp
    # -------------------------
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_type = "incremental" if is_incremental else "fresh"
    output_dir = Path(f"models/{training_type}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"📁 Output directory: {output_dir}")

    with mlflow.start_run(run_name="mbg-sentimen-analysis-model"):

        # Log experiment config
        mlflow.log_params({
            "base_model": BASE_MODEL,
            "model_source": model_source,
            "is_incremental": is_incremental,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "freeze_layers": freeze_layers if is_incremental else 0,
            "batch_size": batch_size,
            "max_length": max_length,
            "weight_decay": weight_decay,
            "validation_split": validation_split,
            "min_f1_macro_threshold": min_f1_macro
        })

        # -------------------------
        # Load and split data
        # -------------------------
        logging.info("📂 Loading training data from: %s", args.train_data)
        full_dataset = load_dataset(args.train_data)

        # Split into train and validation
        from sklearn.model_selection import train_test_split
        train_indices, val_indices = train_test_split(
            range(len(full_dataset)),
            test_size=validation_split,
            random_state=42,
            stratify=full_dataset["label"]  # Stratified split to maintain label distribution
        )

        train_ds_raw = full_dataset.select(train_indices)
        val_ds_raw = full_dataset.select(val_indices)

        logging.info(f"📊 Train size: {len(train_ds_raw)}, Validation size: {len(val_ds_raw)}")


        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        logging.info("📦 Loading model from: %s", model_source)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            num_labels=num_labels
        )

        if is_incremental and freeze_layers > 0:
            freeze_lower_layers(model, freeze_layers)
            logging.info("🔒 Incremental training mode: %d layers frozen", freeze_layers)
        else:
            logging.info("🆕 Fresh training mode: all layers trainable")


        train_ds = tokenize_dataset(train_ds_raw, tokenizer, max_length)
        val_ds = tokenize_dataset(val_ds_raw, tokenizer, max_length)


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


        trainer.train()

        # -------------------------
        # Final evaluation & metrics
        # -------------------------
        metrics = trainer.evaluate()
        logging.info("📊 Evaluation metrics: %s", metrics)

        # Log metrics to MLflow
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

        # -------------------------
        # Model Performance Gate
        # -------------------------
        f1_macro_score = metrics.get("eval_f1_macro", 0)
        
        if f1_macro_score < min_f1_macro:
            logging.warning(f"⚠️ MODEL PERFORMANCE GATE FAILED!")
            logging.warning(f"   F1 Macro: {f1_macro_score:.4f} < Threshold: {min_f1_macro:.4f}")
            logging.warning(f"   Model will NOT be pushed to MLflow registry")
            
            mlflow.set_tag("performance_gate", "FAILED")
            mlflow.set_tag("gate_reason", f"F1 Macro {f1_macro_score:.4f} below threshold {min_f1_macro:.4f}")
            
            push_to_mlflow = False
        else:
            logging.info(f"✅ MODEL PERFORMANCE GATE PASSED!")
            logging.info(f"   F1 Macro: {f1_macro_score:.4f} >= Threshold: {min_f1_macro:.4f}")
            logging.info(f"   Model will be pushed to MLflow")
            
            mlflow.set_tag("performance_gate", "PASSED")
            mlflow.set_tag("gate_reason", f"F1 Macro {f1_macro_score:.4f} meets threshold {min_f1_macro:.4f}")
            
            push_to_mlflow = True

        # -------------------------
        # Generate confusion matrix
        # -------------------------
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.metrics import confusion_matrix

            # Get predictions for confusion matrix
            predictions = trainer.predict(val_ds)
            y_pred = np.argmax(predictions.predictions, axis=1)
            y_true = predictions.label_ids

            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                xticklabels=["Negative", "Neutral", "Positive"],
                yticklabels=["Negative", "Neutral", "Positive"],
                cmap="Blues"
            )
            plt.title("Confusion Matrix")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.tight_layout()

            # Save and log confusion matrix
            cm_path = "confusion_matrix.png"
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
            plt.close()

            # Remove temporary file
            if os.path.exists(cm_path):
                os.remove(cm_path)

            logging.info("✅ Confusion matrix logged to MLflow")
        except Exception as e:
            logging.warning(f"⚠️ Could not generate confusion matrix: {e}")

        # -------------------------
        # Save model locally
        # -------------------------
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        # -------------------------
        # Save metrics summary
        # -------------------------
        metrics_summary = {
            "accuracy": float(metrics.get("eval_accuracy", 0)),
            "f1_macro": float(metrics.get("eval_f1_macro", 0)),
            "loss": float(metrics.get("eval_loss", 0)),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "freeze_layers": freeze_layers if is_incremental else 0,
            "is_incremental": is_incremental,
            "training_type": training_type,
            "performance_gate_passed": push_to_mlflow,
            "performance_gate_threshold": min_f1_macro
        }

        metrics_path = output_dir / "training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_summary, f, indent=2)

        # Log metrics summary as artifact
        mlflow.log_artifact(str(metrics_path))

        # -------------------------
        # Log model artifacts to MLflow (with performance gate)
        # -------------------------
        if push_to_mlflow:
            try:
                # Log all model files as artifacts
                mlflow.log_artifacts(str(output_dir), artifact_path="model")

                logging.info("✅ Model artifacts logged to MLflow")
            except Exception as e:
                logging.warning(f"⚠️ Could not log model artifacts: {e}")
        else:
            logging.info("⏭️ Skipping MLflow artifact logging (performance gate failed)")

        logging.info("💾 Model saved to %s", output_dir)
        
        if push_to_mlflow:
            logging.info("✅ Incremental training finished successfully - Model pushed to MLflow")
        else:
            logging.info("⚠️ Incremental training finished - Model NOT pushed to MLflow (performance gate failed)")