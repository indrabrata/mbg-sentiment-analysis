"""
Logging configuration module for Sentiment Analysis API.
Provides structured logging setup for monitoring and observability.
"""
import logging
import logging.config
from config import LOG_LEVEL, LOG_FORMAT

# Logging configuration dictionary
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s %(request_id)s %(duration_ms)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "json" if LOG_FORMAT == "json" else "standard",
            "stream": "ext://sys.stdout"
        }
    },
    "loggers": {
        "sentiment_api": {
            "level": LOG_LEVEL,
            "handlers": ["console"],
            "propagate": False
        },
        "uvicorn": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"]
    }
}

def setup_logging():
    """
    Initialize logging configuration.
    Falls back to basic logging if python-json-logger is not installed.
    """
    try:
        logging.config.dictConfig(LOGGING_CONFIG)
    except ImportError:
        # Fallback if python-json-logger is not installed
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

def get_logger(name: str = "sentiment_api") -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (default: "sentiment_api")

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
