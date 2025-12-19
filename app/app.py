"""
Main application file for Sentiment Analysis API.
This module initializes the FastAPI application and coordinates all components.
"""
import traceback
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI

from config import API_HOST, API_PORT, API_RELOAD
from logging_config import setup_logging, get_logger
from middleware import RequestLoggingMiddleware
from endpoints import router
from model_loader import load_model
from database import test_connection

# Setup logging
setup_logging()
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """

    logger.info("Application starting up", extra={"event": "startup"})
    try:
        # Test database connection
        logger.info("Testing database connection...", extra={"event": "db_test"})
        if test_connection():
            logger.info("✅ Database connection successful", extra={"event": "db_connected"})
        else:
            logger.warning("⚠️ Database connection failed - predictions won't be saved", extra={"event": "db_not_connected"})

        # Load ML model
        load_model()
        logger.info("Application startup completed successfully", extra={"event": "startup_complete"})
    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}", extra={
            "event": "startup_failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        raise

    yield

    # Shutdown
    logger.info("Application shutting down", extra={"event": "shutdown"})

# Create FastAPI application
app = FastAPI(
    title="Sentiment Analysis API",
    description="IndoBERTweet-based sentiment analysis for Indonesian text",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(RequestLoggingMiddleware)

# Include API routes
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("app:app", host=API_HOST, port=API_PORT, reload=API_RELOAD)
