# ============================================================
# HARD ISOLATION FOR MLFLOW CLIENT ENV
# ============================================================
import os

# --- Force MLflow to use tracking server for artifacts ---
os.environ["MLFLOW_ENABLE_ARTIFACTS"] = "true"

# --- REMOVE ANY S3 / MINIO / AWS CONFIG FROM CLIENT ---
for k in [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_DEFAULT_REGION",
    "MLFLOW_S3_ENDPOINT_URL",
    "MINIO_ROOT_USER",
    "MINIO_ROOT_PASSWORD",
    "MINIO_BUCKET",
]:
    os.environ.pop(k, None)

# ============================================================
# NOW SAFE TO IMPORT MLFLOW
# ============================================================
import argparse
import logging
from pathlib import Path

import mlflow
from mlflow import MlflowClient

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ============================================================
# MLFLOW CLIENT SETUP
# ============================================================
def setup_mlflow_env():
    """
    Client ONLY talks to MLflow server.
    Artifacts are proxied by server to MinIO.
    """

    os.environ["MLFLOW_TRACKING_URI"] = "https://mlops-mlflow.cupcakez.my.id"
    os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    logging.info(f"MLflow Connected To: {mlflow.get_tracking_uri()}")

# ============================================================
# FIRST MODEL UPLOAD
# ============================================================
def upload_first_model(model_dir: str, model_name: str, stage: str | None):
    model_dir = Path(model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    client = MlflowClient()
    EXPERIMENT_NAME = "mbg-sentiment-analysis"
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"register-{model_name}") as run:
        run_id = run.info.run_id
        logging.info(f"Run ID: {run_id}")

        # --- IMPORTANT: log via client (server handles artifact) ---
        client.log_artifacts(
            run_id=run_id,
            local_dir=str(model_dir),
            artifact_path="model"
        )

        model_uri = f"runs:/{run_id}/model"
        logging.info(f"Model URI: {model_uri}")

        mv = client.register_model(
            model_uri=model_uri,
            name=model_name
        )

        logging.info(f"Registered model version: {mv.version}")

        if stage:
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage=stage,
                archive_existing_versions=False
            )
            logging.info(f"Model moved to stage: {stage}")

        logging.info("🎉 FIRST MODEL UPLOAD SUCCESS")
        return mv.version

# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--model_name", default="MBGModel")
    parser.add_argument("--stage", default="Production")
    args = parser.parse_args()

    setup_mlflow_env()
    upload_first_model(
        model_dir=args.model_dir,
        model_name=args.model_name,
        stage=args.stage
    )
