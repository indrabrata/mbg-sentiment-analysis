#!/usr/bin/env python3
"""
pull_latest_model.py
Download latest MLflow model version (Production → Staging) into ./models/previous
"""

import argparse
import os
import shutil
import logging
from mlflow.tracking import MlflowClient
import mlflow

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
    logging.info("Configured MLflow to: %s", mlflow.get_tracking_uri())


# ----------------------------------------------------
# DOWNLOAD MODEL FROM MLFLOW REGISTRY
# ----------------------------------------------------
def download_latest_model(model_name: str, target_dir: str, preferred_stages=("Production", "Staging")):
    client = MlflowClient()
    logging.info("Looking up model '%s' in MLflow Registry", model_name)

    for stage in preferred_stages:
        try:
            versions = client.get_latest_versions(model_name, stages=[stage])
        except Exception as e:
            logging.warning("Failed to fetch versions: %s", e)
            continue

        if versions:
            mv = versions[0]
            run_id = mv.run_id
            logging.info("Selected model version %s (stage=%s)", mv.version, stage)

            dst = os.path.abspath(target_dir)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            os.makedirs(dst, exist_ok=True)

            logging.info("Downloading artifacts from run %s into %s", run_id, dst)
            mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model", dst_path=dst)

            logging.info("Model downloaded to %s", dst)
            return dst

    logging.warning("No model found in preferred stages: %s", preferred_stages)
    return None


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="MBGModel")
    parser.add_argument("--target_dir", default="./models/previous")
    args = parser.parse_args()

    setup_mlflow_env()

    out = download_latest_model(args.model_name, args.target_dir)
    if not out:
        logging.error("Failed to pull model from MLflow registry.")
        raise SystemExit(2)
