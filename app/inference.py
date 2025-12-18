"""
Inference module for Sentiment Analysis API.
Handles prediction logic for single and batch predictions.
"""
import time
import traceback
from typing import Optional, List, Dict
import pandas as pd

from model_loader import get_model
from logging_config import get_logger

logger = get_logger(__name__)

def predict_single(text: str, request_id: Optional[str] = None) -> dict:
    """
    Perform sentiment prediction on a single text.

    Args:
        text: Input text to analyze
        request_id: Optional request ID for logging

    Returns:
        Dictionary with prediction results (label, score)

    Raises:
        Exception: If inference fails
    """
    start_time = time.time()
    model_pipeline = get_model()

    try:
        # Handle different model types (MLflow wrapper vs direct pipeline)
        if hasattr(model_pipeline, 'predict'):
            result = model_pipeline.predict(pd.DataFrame({'text': [text]}))
            if isinstance(result, list):
                result = result[0]
        else:
            result = model_pipeline(text)

        inference_time_ms = (time.time() - start_time) * 1000

        logger.debug("Model inference completed", extra={
            "event": "inference_complete",
            "request_id": request_id,
            "inference_time_ms": round(inference_time_ms, 2),
            "text_length": len(text)
        })

        return result

    except Exception as e:
        logger.error(f"Model inference failed: {str(e)}", extra={
            "event": "inference_failed",
            "request_id": request_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        raise

def predict_batch(texts: List[str], request_id: Optional[str] = None) -> List[dict]:
    """
    Perform sentiment prediction on multiple texts.

    Args:
        texts: List of input texts to analyze
        request_id: Optional request ID for logging

    Returns:
        List of dictionaries with prediction results

    Raises:
        Exception: If inference fails
    """
    start_time = time.time()
    predictions = []

    logger.info(f"Starting batch inference for {len(texts)} texts", extra={
        "event": "batch_inference_start",
        "request_id": request_id,
        "batch_size": len(texts)
    })

    try:
        for idx, text in enumerate(texts):
            result = predict_single(text, request_id)
            predictions.append(result)

        inference_time_ms = (time.time() - start_time) * 1000

        logger.info(f"Batch inference completed for {len(texts)} texts", extra={
            "event": "batch_inference_complete",
            "request_id": request_id,
            "batch_size": len(texts),
            "total_inference_time_ms": round(inference_time_ms, 2),
            "avg_time_per_text_ms": round(inference_time_ms / len(texts), 2)
        })

        return predictions

    except Exception as e:
        logger.error(f"Batch inference failed: {str(e)}", extra={
            "event": "batch_inference_failed",
            "request_id": request_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        raise
