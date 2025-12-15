import argparse
import os
import logging
import mlflow
from mlflow import MlflowClient
from pathlib import Path

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# -------------------------------------------------------------------
# MLflow Setup (CLIENT SIDE)
# -------------------------------------------------------------------
def setup_mlflow_env():
    """
    Client hanya perlu tahu MLflow Tracking Server.
    MinIO di-handle sepenuhnya oleh MLflow server.
    """

    os.environ["MLFLOW_TRACKING_URI"] = "https://mlops-mlflow.cupcakez.my.id"
    os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    logging.info(f"MLflow Connected To: {mlflow.get_tracking_uri()}")

# -------------------------------------------------------------------
# Upload + Register First Model
# -------------------------------------------------------------------
def upload_first_model(model_dir: str, model_name: str, stage: str | None):
    """
    Upload model artifacts, register to Model Registry,
    and optionally move to a stage.
    """

    model_dir = Path(model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    client = MlflowClient()

    EXPERIMENT_NAME = "mbg-sentiment-analysis"
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"register-{model_name}") as run:
        run_id = run.info.run_id
        logging.info(f"Run ID: {run_id}")

        # ------------------------------------------------------------
        # Log artifacts (MODEL FOLDER)
        # ------------------------------------------------------------
        artifact_subdir = "model"
        mlflow.log_artifacts(
            local_dir=str(model_dir),
            artifact_path=artifact_subdir
        )

        model_uri = f"runs:/{run_id}/{artifact_subdir}"
        logging.info(f"Model URI: {model_uri}")

        # ------------------------------------------------------------
        # Register Model
        # ------------------------------------------------------------
        mv = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        version = mv.version
        logging.info(f"Registered Model Version: {version}")

        # ------------------------------------------------------------
        # Transition Stage (Optional)
        # ------------------------------------------------------------
        if stage:
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=False
            )
            logging.info(f"Model moved to stage: {stage}")

        logging.info("🎉 FIRST MODEL UPLOAD SUCCESS")
        return version

# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload and register first model to MLflow"
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to trained model directory (e.g. best_model/)"
    )
    parser.add_argument(
        "--model_name",
        default="MBGModel",
        help="Registered model name in MLflow"
    )
    parser.add_argument(
        "--stage",
        default="Production",
        help="Target stage (Production, Staging, None)"
    )

    args = parser.parse_args()

    setup_mlflow_env()
    upload_first_model(
        model_dir=args.model_dir,
        model_name=args.model_name,
        stage=args.stage
    )
