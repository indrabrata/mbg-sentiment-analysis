#!/usr/bin/env python3
"""
push_to_mlflow.py
Register a local trained model to MLflow Model Registry (Minio as backend).
"""

import argparse
import os
import logging
import mlflow
from mlflow import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ----------------------------------------------------
# SETUP MLFLOW + MINIO ENVIRONMENT
# ----------------------------------------------------
def setup_mlflow_env():
    os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "")
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("MINIO_ACCESS_KEY", "")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("MINIO_SECRET_KEY", "")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MINIO_ENDPOINT", "")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))
    logging.info("MLflow tracking configured at %s", mlflow.get_tracking_uri())


# ----------------------------------------------------
# PUSH MODEL INTO MLFLOW
# ----------------------------------------------------
def push_model(model_dir: str, model_name: str, target_stage: str):
    client = MlflowClient()

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logging.info("MLflow run: %s", run_id)

        mlflow.log_artifacts(model_dir, artifact_path="model")

        model_uri = f"runs:/{run_id}/model"
        logging.info("Registering model from %s", model_uri)

        try:
            mv = mlflow.register_model(model_uri, model_name)
            logging.info("Model registered as version %s", mv.version)
        except Exception as e:
            logging.error("Model registration failed: %s", e)
            raise

        # transition to stage
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage=target_stage,
                archive_existing_versions=False
            )
            logging.info("Transitioned model to stage %s", target_stage)
        except Exception as e:
            logging.warning("Stage transition failed: %s", e)

    return mv.version


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--model_name", default="MBGModel")
    parser.add_argument("--stage", default="Staging")
    args = parser.parse_args()

    setup_mlflow_env()
    ver = push_model(args.model_dir, args.model_name, args.stage)
    logging.info("Model push completed. Version=%s", ver)
