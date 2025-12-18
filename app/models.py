"""
Pydantic models for request/response validation.
"""
from pydantic import BaseModel
from typing import Optional

class TextIn(BaseModel):
    """Input model for single text prediction."""
    text: str

class PredictionOut(BaseModel):
    """Output model for prediction results."""
    label: str
    hf_label: str
    score: float

class ModelInfo(BaseModel):
    """Model information response."""
    model_metadata: dict
    mlflow_config: dict

class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    model_loaded: bool
