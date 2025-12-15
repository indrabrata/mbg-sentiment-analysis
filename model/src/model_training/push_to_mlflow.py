#!/usr/bin/env python3
import argparse
import os
import logging
from pathlib import Path

import mlflow
from mlflow import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def setup_mlflow_env():
    os.environ["MLFLOW_TRACKING_URI"] = os.environ["MLFLOW_TRACKING_URI"]
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["MLFLOW_TRACKING_USERNAME"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["MLFLOW_TRACKING_PASSWORD"]

    os.environ["MLFLOW_ENABLE_ARTIFACTS"] = "true"

    for k in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "MLFLOW_S3_ENDPOINT_URL"]:
        os.environ.pop(k, None)

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


def push_model(model_dir: str, model_name: str, stage: str):
    model_dir = Path(model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)

    client = MlflowClient()

    with mlflow.start_run(run_name=f"train-{model_name}") as run:
        run_id = run.info.run_id

        client.log_artifacts(
            run_id=run_id,
            local_dir=str(model_dir),
            artifact_path="model"
        )

        model_uri = f"runs:/{run_id}/model"
        mv = client.register_model(model_uri, model_name)

        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage=stage,
            archive_existing_versions=False
        )

        logging.info("Model %s version %s → %s", model_name, mv.version, stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--model_name", default="MBGModel")
    parser.add_argument("--stage", default="Staging")
    args = parser.parse_args()

    setup_mlflow_env()
    push_model(args.model_dir, args.model_name, args.stage)
