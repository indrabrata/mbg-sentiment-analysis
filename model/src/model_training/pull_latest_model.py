#!/usr/bin/env python3
import argparse
import os
import shutil
import logging
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def setup_mlflow_env():
    os.environ["MLFLOW_TRACKING_URI"] = os.environ["MLFLOW_TRACKING_URI"]
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["MLFLOW_TRACKING_USERNAME"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["MLFLOW_TRACKING_PASSWORD"]

    # force artifact proxy
    os.environ["MLFLOW_ENABLE_ARTIFACTS"] = "true"

    # hard-clean s3 env
    for k in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "MLFLOW_S3_ENDPOINT_URL"]:
        os.environ.pop(k, None)

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


def download_latest_model(model_name: str, target_dir: str):
    client = MlflowClient()

    for stage in ("Production", "Staging"):
        versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            continue

        mv = versions[0]
        run_id = mv.run_id

        dst = Path(target_dir).resolve()
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)

        logging.info("Downloading model %s (stage=%s)", mv.version, stage)
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="model",
            dst_path=str(dst)
        )
        return str(dst)

    raise RuntimeError("No model found in Production/Staging")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="MBGModel")
    parser.add_argument("--target_dir", default="models/previous")
    args = parser.parse_args()

    setup_mlflow_env()
    download_latest_model(args.model_name, args.target_dir)
