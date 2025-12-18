"""
Database module for PostgreSQL connection and operations.
Handles storing predictions in the database.
"""
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Optional, Dict, Any
from datetime import datetime

from config import (
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    POSTGRES_DB
)
from logging_config import get_logger

logger = get_logger(__name__)

@contextmanager
def get_db_connection():
    """
    Context manager for database connections.
    Automatically handles connection cleanup.
    """
    conn = None
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            dbname=POSTGRES_DB,
            cursor_factory=RealDictCursor
        )
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}", extra={
            "event": "db_error",
            "error": str(e)
        })
        raise
    finally:
        if conn:
            conn.close()


def insert_prediction(
    tweet_data: Dict[str, Any],
    prediction: Dict[str, Any],
    model_run_id: Optional[str] = None
) -> bool:
    """
    Insert a prediction record into the database.

    Args:
        tweet_data: Dictionary containing tweet information
        prediction: Dictionary containing prediction results (label, hf_label, score)
        model_run_id: Optional MLflow run ID used for prediction

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Parse created_at if it's a string
            created_at = tweet_data.get('created_at')
            if isinstance(created_at, str):
                try:
                    # Try to parse common Twitter date format
                    created_at = datetime.strptime(created_at, '%a %b %d %H:%M:%S %z %Y')
                except:
                    created_at = None

            query = """
                INSERT INTO sentiment_analysis_predictions (
                    id_str, conversation_id_str, created_at, tweet_url, lang,
                    full_text, clean_text, user_id_str, username, location,
                    favorite_count, quote_count, reply_count, retweet_count,
                    image_url, in_reply_to_screen_name,
                    pred_label, hf_label, pred_score, original_label,
                    model_run_id, predicted_at
                ) VALUES (
                    %(id_str)s, %(conversation_id_str)s, %(created_at)s, %(tweet_url)s, %(lang)s,
                    %(full_text)s, %(clean_text)s, %(user_id_str)s, %(username)s, %(location)s,
                    %(favorite_count)s, %(quote_count)s, %(reply_count)s, %(retweet_count)s,
                    %(image_url)s, %(in_reply_to_screen_name)s,
                    %(pred_label)s, %(hf_label)s, %(pred_score)s, %(original_label)s,
                    %(model_run_id)s, %(predicted_at)s
                )
                ON CONFLICT (id_str) DO UPDATE SET
                    pred_label = EXCLUDED.pred_label,
                    hf_label = EXCLUDED.hf_label,
                    pred_score = EXCLUDED.pred_score,
                    model_run_id = EXCLUDED.model_run_id,
                    predicted_at = EXCLUDED.predicted_at
            """

            params = {
                'id_str': tweet_data.get('id_str'),
                'conversation_id_str': tweet_data.get('conversation_id_str'),
                'created_at': created_at,
                'tweet_url': tweet_data.get('tweet_url'),
                'lang': tweet_data.get('lang'),
                'full_text': tweet_data.get('full_text'),
                'clean_text': tweet_data.get('clean_text'),
                'user_id_str': tweet_data.get('user_id_str'),
                'username': tweet_data.get('username'),
                'location': tweet_data.get('location'),
                'favorite_count': tweet_data.get('favorite_count', 0),
                'quote_count': tweet_data.get('quote_count', 0),
                'reply_count': tweet_data.get('reply_count', 0),
                'retweet_count': tweet_data.get('retweet_count', 0),
                'image_url': tweet_data.get('image_url'),
                'in_reply_to_screen_name': tweet_data.get('in_reply_to_screen_name'),
                'pred_label': prediction['label'],
                'hf_label': prediction['hf_label'],
                'pred_score': prediction['score'],
                'original_label': tweet_data.get('label'),  # Original label if available
                'model_run_id': model_run_id,
                'predicted_at': datetime.now()
            }

            cursor.execute(query, params)

            logger.info(f"Prediction saved to database", extra={
                "event": "prediction_saved",
                "id_str": tweet_data.get('id_str'),
                "pred_label": prediction['label'],
                "pred_score": prediction['score']
            })

            return True

    except Exception as e:
        logger.error(f"Failed to insert prediction: {e}", extra={
            "event": "prediction_insert_failed",
            "error": str(e),
            "id_str": tweet_data.get('id_str')
        })
        return False


def test_connection() -> bool:
    """
    Test database connection.

    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            logger.info("Database connection test successful", extra={
                "event": "db_test_success"
            })
            return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}", extra={
            "event": "db_test_failed",
            "error": str(e)
        })
        return False
