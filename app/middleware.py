"""
Middleware module for Sentiment Analysis API.
Handles request logging and tracking.
"""
import time
import uuid
import traceback
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from logging_config import get_logger

logger = get_logger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all HTTP requests and responses.
    Adds request ID for tracing and tracks request duration.
    """

    async def dispatch(self, request: Request, call_next):
        """
        Process the request and log details.

        Args:
            request: The incoming HTTP request
            call_next: The next middleware or route handler

        Returns:
            HTTP response with added request ID header
        """
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.time()

        # Log incoming request
        logger.info(f"Incoming request: {request.method} {request.url.path}", extra={
            "event": "request_start",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client_host": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent", "unknown")
        })

        try:
            # Process request
            response = await call_next(request)
            duration_ms = (time.time() - start_time) * 1000

            # Log successful response
            logger.info(f"Request completed: {request.method} {request.url.path}", extra={
                "event": "request_complete",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2)
            })

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            return response

        except Exception as e:
            # Log failed request
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Request failed: {request.method} {request.url.path}", extra={
                "event": "request_failed",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "error": str(e),
                "duration_ms": round(duration_ms, 2),
                "traceback": traceback.format_exc()
            })
            raise
