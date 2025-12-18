"""
Model loading module for Sentiment Analysis API.
Handles loading models from MLflow registry or runs.
"""
import traceback
import tempfile
from datetime import datetime
import mlflow
import mlflow.pyfunc
import mlflow.transformers
from transformers import pipeline as hf_pipeline

from config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_NAME,
    MODEL_STAGE,
    MODEL_METADATA
)
from logging_config import get_logger

logger = get_logger(__name__)

# Global variable to store the loaded model
MODEL_PIPELINE = None

def get_model():
    """Get the loaded model pipeline."""
    global MODEL_PIPELINE
    if MODEL_PIPELINE is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return MODEL_PIPELINE

def load_model():
    """
    Load the model from MLflow registry or latest run.

    Raises:
        RuntimeError: If model loading fails
    """
    global MODEL_PIPELINE
    MODEL_METADATA["loaded_at"] = datetime.now().isoformat()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"Connecting to MLflow at: {MLFLOW_TRACKING_URI}", extra={
        "event": "mlflow_connect",
        "tracking_uri": MLFLOW_TRACKING_URI
    })

    # Try loading from registry if MODEL_NAME is provided
    if MODEL_NAME:
        MODEL_PIPELINE = _load_from_registry()
        return

    # Otherwise, load from latest run
    MODEL_PIPELINE = _load_from_latest_run()

def _load_from_registry():
    """
    Load model from MLflow registry.

    Returns:
        Loaded model pipeline

    Raises:
        RuntimeError: If loading fails
    """
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        logger.info(f"Loading model from registry: {model_uri}", extra={
            "event": "model_load_start",
            "source": "registry",
            "model_name": MODEL_NAME,
            "model_stage": MODEL_STAGE,
            "model_uri": model_uri
        })

        pipeline = mlflow.pyfunc.load_model(model_uri)
        MODEL_METADATA["source"] = "mlflow_registry"
        MODEL_METADATA["model_name"] = MODEL_NAME
        MODEL_METADATA["stage"] = MODEL_STAGE

        logger.info(f"Model loaded from MLflow registry: {MODEL_NAME}/{MODEL_STAGE}", extra={
            "event": "model_load_success",
            "source": "registry",
            "model_name": MODEL_NAME,
            "model_stage": MODEL_STAGE
        })

        return pipeline

    except Exception as e:
        error_msg = f"Failed to load from registry: {e}"
        logger.error(error_msg, extra={
            "event": "model_load_failed",
            "source": "registry",
            "model_name": MODEL_NAME,
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        raise RuntimeError(error_msg)

def _load_from_latest_run():
    """
    Load model from the latest MLflow run.

    Returns:
        Loaded model pipeline

    Raises:
        RuntimeError: If loading fails
    """
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

        if not experiment:
            error_msg = f"Experiment '{MLFLOW_EXPERIMENT_NAME}' not found in MLflow"
            logger.error(error_msg, extra={
                "event": "experiment_not_found",
                "experiment_name": MLFLOW_EXPERIMENT_NAME
            })
            raise RuntimeError(error_msg)

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="",
            order_by=["start_time DESC"],
            max_results=1
        )

        if not runs:
            error_msg = f"No runs found in experiment '{MLFLOW_EXPERIMENT_NAME}'"
            logger.error(error_msg, extra={
                "event": "no_runs_found",
                "experiment_name": MLFLOW_EXPERIMENT_NAME
            })
            raise RuntimeError(error_msg)

        latest_run = runs[0]
        run_id = latest_run.info.run_id
        model_uri = f"runs:/{run_id}/model"

        logger.info(f"Loading model from latest run: {run_id}", extra={
            "event": "model_load_start",
            "source": "run",
            "run_id": run_id,
            "experiment_name": MLFLOW_EXPERIMENT_NAME
        })

        # Download artifacts
        logger.info("Downloading model artifacts...", extra={
            "event": "artifacts_download_start",
            "run_id": run_id
        })

        temp_dir = tempfile.mkdtemp()
        model_path = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri,
            dst_path=temp_dir
        )

        logger.info(f"Artifacts downloaded to: {model_path}", extra={
            "event": "artifacts_download_success",
            "model_path": model_path,
            "run_id": run_id
        })

        # Load the model directly using transformers pipeline
        pipeline = hf_pipeline("text-classification", model=model_path)

        MODEL_METADATA["source"] = "mlflow_run"
        MODEL_METADATA["run_id"] = run_id
        MODEL_METADATA["experiment_name"] = MLFLOW_EXPERIMENT_NAME

        logger.info(f"Model loaded from MLflow run: {run_id}", extra={
            "event": "model_load_success",
            "source": "run",
            "run_id": run_id,
            "experiment_name": MLFLOW_EXPERIMENT_NAME
        })

        return pipeline

    except Exception as e:
        error_msg = f"Failed to load model from MLflow: {str(e)}"
        logger.error(error_msg, extra={
            "event": "model_load_failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        raise RuntimeError(error_msg)
