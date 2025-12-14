import argparse
import os
import logging
import shutil
import mlflow
from mlflow import MlflowClient
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def setup_mlflow_env():
    os.environ["MLFLOW_TRACKING_URI"] = "https://mlops-mlflow.cupcakez.my.id"
    os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://mlops-minio-api.cupcakez.my.id"
    os.environ["MLFLOW_S3_IGNORE_SSL"] = "true"
    os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"

    mlflow.set_tracking_uri("https://mlops-mlflow.cupcakez.my.id")
    logging.info(f"MLflow Connected To: {mlflow.get_tracking_uri()}")


def upload_first_model(model_dir: str, model_name: str, stage: str):
    client = MlflowClient()

    EXPERIMENT = "mbg-sentiment-analysis"
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logging.info(f"Run ID: {run_id}")

        # Copy best_model/ → artifact folder
        artifact_subdir = "model"
        mlflow.log_artifacts(model_dir, artifact_path=artifact_subdir)

        model_uri = f"runs:/{run_id}/{artifact_subdir}"
        logging.info(f"Model URI: {model_uri}")

        # Register model
        mv = mlflow.register_model(model_uri, model_name)
        version = mv.version
        logging.info(f"Registered version: {version}")

        if stage:
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=False
            )
            logging.info(f"Model moved to stage: {stage}")

        logging.info("🎉 MODEL UPLOAD SUCCESS!")
        return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--model_name", default="MBGModel")
    parser.add_argument("--stage", default="Production")
    args = parser.parse_args()

    setup_mlflow_env()
    upload_first_model(args.model_dir, args.model_name, args.stage)
