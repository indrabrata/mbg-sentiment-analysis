import os
import shutil
import logging
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def setup_mlflow_env():
    """Setup MLflow environment variables from .env file"""
    # Ensure required env vars are set
    required_vars = ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]
    for var in required_vars:
        if var not in os.environ:
            raise EnvironmentError(f"Missing required environment variable: {var}")

    # Force artifact proxy
    if "MLFLOW_ENABLE_ARTIFACTS" not in os.environ:
        os.environ["MLFLOW_ENABLE_ARTIFACTS"] = "true"

    # Clean S3 environment variables to avoid conflicts
    for k in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "MLFLOW_S3_ENDPOINT_URL"]:
        os.environ.pop(k, None)

    # Set MLflow tracking URI
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
