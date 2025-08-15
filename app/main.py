"""Main FastAPI application."""
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import time
import structlog
from datetime import datetime
from loguru import logger
import sys
from pathlib import Path
import asyncio

from app.core.config import settings
from app.api.v1 import api_router
from app.core.database import create_tables

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Configure loguru
logger.remove()
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    level=settings.LOG_LEVEL,
    colorize=True
)
logger.add(
    "logs/app.log",
    rotation="1 day",
    retention="30 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    level=settings.LOG_LEVEL
)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-level FastAPI backend for RTSP camera management with facial recognition",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and responses."""
    start_time = time.time()
    
    # Log request
    logger.info(
        f"Request: {request.method} {request.url} - Client: {request.client.host if request.client else 'unknown'}"
    )
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log response
    logger.info(
        f"Response: {request.method} {request.url} - Status: {response.status_code} - Time: {process_time:.3f}s"
    )
    
    # Add processing time header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Validation error",
            "errors": exc.errors()
        }
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    logger.error(f"HTTP error: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "status_code": 500
        }
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        from app.core.database import check_db_health, get_db_stats
        
        # Check database health
        db_healthy = check_db_health()
        db_stats = get_db_stats()
        
        return {
            "status": "healthy" if db_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": "healthy" if db_healthy else "unhealthy",
            "database_stats": db_stats
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": "unhealthy",
            "error": str(e)
        }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": "/docs" if settings.DEBUG else None,
        "health": "/health"
    }


# Mount static files for HLS streams
uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)
streams_dir = uploads_dir / "streams"
streams_dir.mkdir(exist_ok=True)
# app.mount("/streams", StaticFiles(directory="uploads/streams"), name="streams")

@app.get("/health/database")
async def check_database_health_detailed():
    """Detailed database health check with connection pool information."""
    try:
        from app.core.database import check_db_health, get_db_stats, reset_db_connections
        
        db_healthy = check_db_health()
        db_stats = get_db_stats()
        
        return {
            "status": "healthy" if db_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database_health": db_healthy,
            "connection_pool_stats": db_stats,
            "recommendations": _get_db_recommendations(db_stats)
        }
        
    except Exception as e:
        logger.error(f"Detailed database health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@app.post("/database/reset-connections")
async def reset_database_connections():
    """Reset all database connections in the pool (use for debugging)."""
    try:
        from app.core.database import reset_db_connections
        
        success = reset_db_connections()
        
        if success:
            return {
                "success": True,
                "message": "Database connection pool reset successfully",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "success": False,
                "message": "Failed to reset database connections",
                "timestamp": datetime.utcnow().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Failed to reset database connections: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


def _get_db_recommendations(db_stats: dict) -> list:
    """Get recommendations based on database connection pool statistics."""
    recommendations = []
    
    try:
        if "error" in db_stats:
            recommendations.append("Database engine not available - check configuration")
            return recommendations
        
        pool_size = db_stats.get("pool_size", 0)
        checked_out = db_stats.get("checked_out", 0)
        overflow = db_stats.get("overflow", 0)
        invalid = db_stats.get("invalid", 0)
        
        # Check for connection pool exhaustion
        if checked_out >= pool_size * 0.8:
            recommendations.append("Connection pool usage is high - consider increasing pool size")
        
        # Check for overflow connections
        if overflow > 0:
            recommendations.append(f"Using {overflow} overflow connections - consider increasing pool size")
        
        # Check for invalid connections
        if invalid > 0:
            recommendations.append(f"Found {invalid} invalid connections - consider resetting pool")
        
        # Check pool efficiency
        if pool_size > 0:
            utilization = (checked_out / pool_size) * 100
            if utilization < 20:
                recommendations.append("Connection pool utilization is low - consider reducing pool size")
            elif utilization > 80:
                recommendations.append("Connection pool utilization is high - consider increasing pool size")
        
        if not recommendations:
            recommendations.append("Connection pool appears healthy")
            
    except Exception as e:
        recommendations.append(f"Error analyzing pool stats: {str(e)}")
    
    return recommendations


@app.get("/stream/{camera_id}/playlist.m3u8")
async def get_hls_playlist(camera_id: str):
    """Get HLS playlist for a camera stream."""
    try:
        import uuid
        from pathlib import Path
        
        camera_uuid = uuid.UUID(camera_id)
        playlist_path = Path(f"uploads/streams/{camera_uuid}/playlist.m3u8")
        
        if playlist_path.exists():
            return FileResponse(
                playlist_path,
                media_type="application/vnd.apple.mpegurl",
                headers={"Cache-Control": "no-cache"}
            )
        else:
            return JSONResponse(
                status_code=404,
                content={"error": "Stream not found or not active"}
            )
            
    except Exception as e:
        logger.error(f"Error serving HLS playlist for camera {camera_id}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# Include API routers
app.include_router(
    api_router,
    prefix=settings.API_V1_STR
)


# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    try:
        # Test database connection before proceeding
        from app.core.database import check_db_health, get_db_stats
        
        logger.info("Testing database connection...")
        if not check_db_health():
            logger.error("Database connection test failed during startup")
            raise Exception("Database connection test failed")
        
        # Log database pool statistics
        db_stats = get_db_stats()
        logger.info(f"Database connection pool initialized: {db_stats}")
        
        # Create database tables
        create_tables()
        logger.info("Database tables created/verified successfully")
        
        # Initialize face recognition service
        from app.services.face_recognition.face_service import face_recognition_service
        logger.info("Face recognition service initialized")
        
        logger.info(f"{settings.APP_NAME} started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        # Log detailed error information
        import traceback
        logger.error(f"Startup error details: {traceback.format_exc()}")
        raise


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info(f"Shutting down {settings.APP_NAME}")
    
    try:
        # Stop all camera streams
        from app.services.streaming.stream_service import stream_manager
        await stream_manager.stop_all_streams()
        
        logger.info("All camera streams stopped")
        logger.info(f"{settings.APP_NAME} shut down successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS if not settings.DEBUG else 1,
        log_level=settings.LOG_LEVEL.lower()
    )
