"""
API endpoints for Sentiment Analysis API.
Defines all HTTP routes and handlers.
"""
import traceback
from datetime import datetime
from pathlib import Path
import pandas as pd

from fastapi import APIRouter, HTTPException, UploadFile, File, Request
from fastapi.responses import FileResponse

from models import TextIn, PredictionOut, ModelInfo, HealthCheck
from inference import predict_single
from config import (
    LABEL_MAP,
    MODEL_METADATA,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_NAME,
    MODEL_STAGE,
    PREDICTIONS_OUTPUT_DIR
)
from logging_config import get_logger

logger = get_logger(__name__)

# Create API router
router = APIRouter()

@router.post("/predict", response_model=PredictionOut)
def predict(inp: TextIn, request: Request):
    """
    Perform sentiment analysis on a single text.

    Args:
        inp: Input text data
        request: FastAPI request object

    Returns:
        Prediction result with label and confidence score

    Raises:
        HTTPException: If prediction fails
    """
    request_id = getattr(request.state, 'request_id', 'unknown')

    try:
        logger.info("Starting prediction", extra={
            "event": "prediction_start",
            "request_id": request_id,
            "text_length": len(inp.text)
        })

        res = predict_single(inp.text, request_id)

        if isinstance(res, list) and len(res) > 0:
            hf_label = res[0].get("label")
            score = float(res[0].get("score"))
            mapped_label = LABEL_MAP.get(hf_label, hf_label)

            logger.info("Prediction completed successfully", extra={
                "event": "prediction_success",
                "request_id": request_id,
                "predicted_label": mapped_label,
                "confidence_score": round(score, 4),
                "hf_label": hf_label
            })

            return {
                "label": mapped_label,
                "hf_label": hf_label,
                "score": score
            }
        else:
            error_msg = "Model returned unexpected format"
            logger.error(error_msg, extra={
                "event": "prediction_invalid_format",
                "request_id": request_id,
                "result": str(res)
            })
            raise HTTPException(status_code=500, detail=error_msg)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", extra={
            "event": "prediction_failed",
            "request_id": request_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict-batch")
def predict_batch(request: Request, file: UploadFile = File(...)):
    """
    Perform sentiment analysis on multiple texts from a CSV file.

    Args:
        file: CSV file containing texts to analyze
        request: FastAPI request object

    Returns:
        CSV file with predictions

    Raises:
        HTTPException: If batch prediction fails
    """
    request_id = getattr(request.state, 'request_id', 'unknown')

    logger.info("Starting batch prediction", extra={
        "event": "batch_prediction_start",
        "request_id": request_id,
        "filename": file.filename
    })

    try:
        df = pd.read_csv(file.file)
        logger.info(f"CSV file read successfully: {len(df)} rows", extra={
            "event": "csv_read_success",
            "request_id": request_id,
            "row_count": len(df),
            "filename": file.filename
        })
    except Exception as e:
        logger.error(f"CSV read error: {str(e)}", extra={
            "event": "csv_read_failed",
            "request_id": request_id,
            "error": str(e),
            "filename": file.filename
        })
        raise HTTPException(status_code=400, detail=f"CSV read error: {e}")

    if "text" not in df.columns:
        error_msg = "CSV must contain 'text' column"
        logger.error(error_msg, extra={
            "event": "csv_validation_failed",
            "request_id": request_id,
            "columns": list(df.columns)
        })
        raise HTTPException(status_code=400, detail=error_msg)

    # Perform predictions
    import time
    start_time = time.time()
    preds = []

    for text in df["text"].tolist():
        result = predict_single(text, request_id)
        if isinstance(result, list) and len(result) > 0:
            preds.append(result[0])
        else:
            preds.append(result)

    inference_time = (time.time() - start_time) * 1000

    # Add predictions to dataframe
    df["hf_label"] = [p["label"] for p in preds]
    df["pred_label"] = [LABEL_MAP.get(p["label"], p["label"]) for p in preds]
    df["pred_score"] = [p["score"] for p in preds]

    # Save results
    out_path = Path(PREDICTIONS_OUTPUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)
    save_path = out_path / "predictions_batch.csv"
    df.to_csv(save_path, index=False)

    # Log prediction statistics
    label_counts = df["pred_label"].value_counts().to_dict()
    avg_score = df["pred_score"].mean()

    logger.info("Batch prediction completed successfully", extra={
        "event": "batch_prediction_success",
        "request_id": request_id,
        "total_predictions": len(df),
        "inference_time_ms": round(inference_time, 2),
        "avg_time_per_prediction_ms": round(inference_time / len(df), 2),
        "label_distribution": label_counts,
        "average_confidence_score": round(avg_score, 4),
        "output_file": str(save_path)
    })

    return FileResponse(save_path)

@router.get("/model_info", response_model=ModelInfo)
def get_model_info():
    """
    Get information about the currently loaded model.

    Returns:
        Model metadata and MLflow configuration
    """
    logger.info("Model info requested", extra={"event": "model_info_request"})
    return {
        "model_metadata": MODEL_METADATA,
        "mlflow_config": {
            "tracking_uri": MLFLOW_TRACKING_URI,
            "experiment_name": MLFLOW_EXPERIMENT_NAME,
            "model_name": MODEL_NAME,
            "model_stage": MODEL_STAGE
        }
    }

@router.get("/health", response_model=HealthCheck)
def health_check():
    """
    Health check endpoint for monitoring.

    Returns:
        Application health status
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": MODEL_METADATA.get("loaded_at") is not None
    }
