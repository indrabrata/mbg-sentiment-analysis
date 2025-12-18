"""
Pydantic models for request/response validation.
"""
from pydantic import BaseModel
from typing import Optional

class TextIn(BaseModel):
    """Input model for single text prediction."""
    text: str

class TweetIn(BaseModel):
    """Input model for tweet data with prediction."""
    id_str: str
    clean_text: str
    full_text: Optional[str] = None
    conversation_id_str: Optional[str] = None
    created_at: Optional[str] = None
    tweet_url: Optional[str] = None
    lang: Optional[str] = None
    user_id_str: Optional[str] = None
    username: Optional[str] = None
    location: Optional[str] = None
    favorite_count: Optional[int] = 0
    quote_count: Optional[int] = 0
    reply_count: Optional[int] = 0
    retweet_count: Optional[int] = 0
    image_url: Optional[str] = None
    in_reply_to_screen_name: Optional[str] = None
    label: Optional[str] = None  # Original label if from training data

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
