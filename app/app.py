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

# Setup logging
setup_logging()
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Application starting up", extra={"event": "startup"})
    try:
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
