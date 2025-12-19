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

from models import TextIn, TweetIn, PredictionOut, ModelInfo, HealthCheck
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
from database import insert_prediction

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

@router.post("/predict-tweet", response_model=PredictionOut)
def predict_tweet(tweet: TweetIn, request: Request):
    """
    Perform sentiment analysis on a tweet and store in database.

    Args:
        tweet: Tweet data with clean_text and metadata
        request: FastAPI request object

    Returns:
        Prediction result with label and confidence score

    Raises:
        HTTPException: If prediction or database insertion fails
    """
    request_id = getattr(request.state, 'request_id', 'unknown')

    try:
        logger.info("Starting tweet prediction", extra={
            "event": "tweet_prediction_start",
            "request_id": request_id,
            "id_str": tweet.id_str,
            "text_length": len(tweet.clean_text)
        })

        # Perform prediction on clean_text
        res = predict_single(tweet.clean_text, request_id)

        if isinstance(res, list) and len(res) > 0:
            hf_label = res[0].get("label")
            score = float(res[0].get("score"))
            mapped_label = LABEL_MAP.get(hf_label, hf_label)

            prediction = {
                "label": mapped_label,
                "hf_label": hf_label,
                "score": score
            }

            # Store in database
            try:
                model_run_id = MODEL_METADATA.get("run_id")
                tweet_data = tweet.model_dump()
                insert_prediction(tweet_data, prediction, model_run_id)

                logger.info("Tweet prediction completed and saved", extra={
                    "event": "tweet_prediction_success",
                    "request_id": request_id,
                    "id_str": tweet.id_str,
                    "predicted_label": mapped_label,
                    "confidence_score": round(score, 4),
                    "saved_to_db": True
                })
            except Exception as db_error:
                logger.warning(f"Prediction succeeded but database save failed: {db_error}", extra={
                    "event": "db_save_failed",
                    "request_id": request_id,
                    "id_str": tweet.id_str,
                    "error": str(db_error)
                })
                # Continue even if database save fails

            return prediction
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
        logger.error(f"Tweet prediction failed: {str(e)}", extra={
            "event": "tweet_prediction_failed",
            "request_id": request_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict-batch-tweet")
def predict_batch_tweet(request: Request, file: UploadFile = File(...)):
    """
    Perform sentiment analysis on multiple tweets from CSV and store in database.

    Args:
        file: CSV file with tweet data
        request: FastAPI request object

    Expected CSV columns:
        id_str, conversation_id_str, created_at, full_text, clean_text,
        user_id_str, username, lang, location, favorite_count, quote_count,
        reply_count, retweet_count, image_url, in_reply_to_screen_name, tweet_url

    Returns:
        CSV file with predictions added

    Raises:
        HTTPException: If batch prediction fails
    """
    request_id = getattr(request.state, 'request_id', 'unknown')

    logger.info("Starting batch tweet prediction", extra={
        "event": "batch_tweet_prediction_start",
        "request_id": request_id,
        "uploaded_filename": file.filename
    })

    try:
        # Read CSV file
        df = pd.read_csv(file.file)
        logger.info(f"CSV file read successfully: {len(df)} rows", extra={
            "event": "csv_read_success",
            "request_id": request_id,
            "row_count": len(df),
            "uploaded_filename": file.filename
        })
    except Exception as e:
        logger.error(f"CSV read error: {str(e)}", extra={
            "event": "csv_read_failed",
            "request_id": request_id,
            "error": str(e),
            "uploaded_filename": file.filename
        })
        raise HTTPException(status_code=400, detail=f"CSV read error: {e}")

    # Validate required columns
    required_columns = ["id_str", "clean_text"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        error_msg = f"CSV missing required columns: {', '.join(missing_columns)}"
        logger.error(error_msg, extra={
            "event": "csv_validation_failed",
            "request_id": request_id,
            "missing_columns": missing_columns,
            "available_columns": list(df.columns)
        })
        raise HTTPException(status_code=400, detail=error_msg)

    # Perform predictions and save to database
    import time
    start_time = time.time()
    preds = []
    saved_count = 0
    failed_count = 0

    model_run_id = MODEL_METADATA.get("run_id")

    for idx, row in df.iterrows():
        try:
            # Perform prediction
            result = predict_single(row["clean_text"], request_id)

            if isinstance(result, list) and len(result) > 0:
                hf_label = result[0]["label"]
                score = float(result[0]["score"])
                mapped_label = LABEL_MAP.get(hf_label, hf_label)

                prediction = {
                    "label": mapped_label,
                    "hf_label": hf_label,
                    "score": score
                }
                preds.append(prediction)

                # Save to database
                try:
                    tweet_data = row.to_dict()
                    if insert_prediction(tweet_data, prediction, model_run_id):
                        saved_count += 1
                except Exception as db_error:
                    failed_count += 1
                    logger.warning(f"Failed to save prediction for id_str={row['id_str']}: {db_error}", extra={
                        "event": "db_save_failed",
                        "request_id": request_id,
                        "id_str": row["id_str"],
                        "error": str(db_error)
                    })
            else:
                preds.append({"label": "error", "hf_label": "error", "score": 0.0})
                failed_count += 1

        except Exception as e:
            logger.error(f"Prediction failed for row {idx}: {e}", extra={
                "event": "prediction_failed",
                "request_id": request_id,
                "row_index": idx,
                "error": str(e)
            })
            preds.append({"label": "error", "hf_label": "error", "score": 0.0})
            failed_count += 1

    inference_time = (time.time() - start_time) * 1000

    # Add predictions to dataframe
    df["hf_label"] = [p["hf_label"] for p in preds]
    df["pred_label"] = [p["label"] for p in preds]
    df["pred_score"] = [p["score"] for p in preds]

    # Save results
    out_path = Path(PREDICTIONS_OUTPUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = out_path / f"predictions_batch_tweet_{timestamp}.csv"
    df.to_csv(save_path, index=False)

    # Log statistics
    label_counts = df["pred_label"].value_counts().to_dict()
    avg_score = df[df["pred_score"] > 0]["pred_score"].mean() if len(df[df["pred_score"] > 0]) > 0 else 0

    logger.info("Batch tweet prediction completed successfully", extra={
        "event": "batch_tweet_prediction_success",
        "request_id": request_id,
        "total_predictions": len(df),
        "saved_to_db": saved_count,
        "failed_to_save": failed_count,
        "inference_time_ms": round(inference_time, 2),
        "avg_time_per_prediction_ms": round(inference_time / len(df), 2),
        "label_distribution": label_counts,
        "average_confidence_score": round(avg_score, 4),
        "output_file": str(save_path)
    })

    return FileResponse(save_path)

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
        "uploaded_filename": file.filename
    })

    try:
        df = pd.read_csv(file.file)
        logger.info(f"CSV file read successfully: {len(df)} rows", extra={
            "event": "csv_read_success",
            "request_id": request_id,
            "row_count": len(df),
            "uploaded_filename": file.filename
        })
    except Exception as e:
        logger.error(f"CSV read error: {str(e)}", extra={
            "event": "csv_read_failed",
            "request_id": request_id,
            "error": str(e),
            "uploaded_filename": file.filename
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
