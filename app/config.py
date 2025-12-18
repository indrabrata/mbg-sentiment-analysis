"""
Configuration module for Sentiment Analysis API.
Handles all environment variables and application settings.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Configure MLflow authentication
if os.getenv("MLFLOW_TRACKING_USERNAME"):
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
if os.getenv("MLFLOW_TRACKING_PASSWORD"):
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# Configure AWS/S3 credentials for MLflow artifact storage
if os.getenv("AWS_ACCESS_KEY_ID"):
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
if os.getenv("AWS_SECRET_ACCESS_KEY"):
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
if os.getenv("MLFLOW_S3_ENDPOINT_URL"):
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "mbg-sentiment-analysis")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", None)
MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("API_RELOAD", "false").lower() == "true"

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "json")

# Prediction Configuration
PREDICTIONS_OUTPUT_DIR = os.getenv("PREDICTIONS_OUTPUT_DIR", "data/predictions")

# Label Mapping
LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

# Model Metadata (will be populated at runtime)
MODEL_METADATA = {
    "source": "unknown",
    "run_id": None,
    "model_name": None,
    "loaded_at": None
}
