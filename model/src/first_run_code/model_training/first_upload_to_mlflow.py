#!/usr/bin/env python3
"""
first_upload_to_mlflow.py
Upload FIRST MODEL to MLflow Registry and set it as Production/Staging.
Use this ONCE before enabling weekly retraining pipeline.
"""

import argparse
import os
import logging
import mlflow
from mlflow import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------------------------------------
# SETUP ENV (MLflow + MinIO)
# -------------------------------------------------
def setup_mlflow_env():
    os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "")
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("MINIO_ACCESS_KEY", "")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("MINIO_SECRET_KEY", "")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MINIO_ENDPOINT", "")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    logging.info(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")


# -------------------------------------------------
# FIRST MODEL UPLOAD
# -------------------------------------------------
def upload_first_model(model_dir: str, model_name: str, stage: str):
    client = MlflowClient()

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logging.info(f"MLflow Run ID: {run_id}")

        # Log entire model directory as artifact
        mlflow.log_artifacts(model_dir, artifact_path="model")
        model_uri = f"runs:/{run_id}/model"

        logging.info(f"Registering model: {model_name}")
        mv = mlflow.register_model(model_uri, model_name)
        logging.info(f"✔ Registered version: {mv.version}")

        # Transition version to stage
        if stage:
            logging.info(f"Transitioning model version {mv.version} to stage: {stage}")
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage=stage,
                archive_existing_versions=False,
            )

        logging.info("🎉 FIRST MODEL UPLOAD COMPLETE!")
        return mv.version


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload FIRST trained model to MLflow registry.")
    parser.add_argument("--model_dir", required=True, help="Directory containing best_model/ folder")
    parser.add_argument("--model_name", default="MBGModel")
    parser.add_argument("--stage", default="Production", help="Target stage: Production/Staging")
    args = parser.parse_args()

    setup_mlflow_env()
    upload_first_model(args.model_dir, args.model_name, args.stage)
