"""
MIA (Mobile İSG Aracı) - Backend API
=====================================

FastAPI backend for construction safety violation detection.

Features:
- Model loaded once at startup (lifespan management)
- Async video processing (non-blocking)
- WebSocket real-time streaming
- SQLite violation logging
- Health monitoring

Author: MIA AI Team
Version: 1.0.0

Usage:
    # Development
    uvicorn backend_api.main:app --reload --host 0.0.0.0 --port 8000
    
    # Production
    gunicorn backend_api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import routers
from backend_api.api.v1.router import api_router
from backend_api.api.websocket.stream_handler import websocket_router
from backend_api.core.config import settings
from backend_api.core.state import AppState
from backend_api.db.session import init_db, close_db


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging() -> logging.Logger:
    """Configure application logging."""
    # Create logs directory
    log_dir = Path(settings.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "api.log")
        ]
    )
    
    # Get application logger
    logger = logging.getLogger("mia.api")
    logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
    
    return logger

logger = setup_logging()


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION STATE (Singleton)
# ═══════════════════════════════════════════════════════════════════════════════

# Global application state - holds the AI model
app_state = AppState()


# ═══════════════════════════════════════════════════════════════════════════════
# LIFESPAN MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events:
    - Startup: Load AI model, initialize database
    - Shutdown: Cleanup resources
    
    This ensures the model is loaded ONCE when the server starts,
    not for every request, saving significant RAM and startup time.
    """
    # ─────────────────────────────────────────────────────────────────────────
    # STARTUP
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("🚀 MIA Backend API Starting...")
    logger.info("=" * 60)
    
    try:
        # Initialize database
        logger.info("📦 Initializing database...")
        await init_db()
        logger.info("   Database ready ✓")
        
        # Load AI model
        logger.info("🤖 Loading AI model...")
        await app_state.load_model(settings.MODEL_PATH)
        logger.info(f"   Model loaded: {settings.MODEL_PATH} ✓")
        
        # Create output directories
        Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        Path(settings.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        logger.info("   Directories initialized ✓")
        
        logger.info("=" * 60)
        logger.info("✅ MIA Backend API Ready!")
        logger.info(f"   API Docs: http://{settings.HOST}:{settings.PORT}/docs")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    # ─────────────────────────────────────────────────────────────────────────
    # YIELD CONTROL TO APPLICATION
    # ─────────────────────────────────────────────────────────────────────────
    yield
    
    # ─────────────────────────────────────────────────────────────────────────
    # SHUTDOWN
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("🛑 MIA Backend API Shutting down...")
    
    # Cleanup
    await app_state.unload_model()
    await close_db()
    
    logger.info("👋 Shutdown complete")


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="MIA - Mobile İSG Aracı",
    description="""
    ## Construction Safety Violation Detection API
    
    Real-time PPE (Personal Protective Equipment) detection for construction sites.
    
    ### Features
    - 🎥 **Video Upload**: Process video files for violation detection
    - 📡 **WebSocket Stream**: Real-time video stream processing
    - 📊 **Violation Logs**: Track all detected safety violations
    - 🔍 **Health Monitoring**: API status and model health checks
    
    ### Detected Violations
    - Missing Helmet (no_helmet)
    - Missing Safety Vest (no_vest)
    - Missing Harness (no_harness)
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Store app state in app.state for access in routes
app.state.app_state = app_state


# ═══════════════════════════════════════════════════════════════════════════════
# MIDDLEWARE
# ═══════════════════════════════════════════════════════════════════════════════

# CORS Middleware - Allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = datetime.now()
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = (datetime.now() - start_time).total_seconds() * 1000
    
    # Log request
    logger.info(
        f"{request.method} {request.url.path} "
        f"| Status: {response.status_code} "
        f"| Duration: {duration:.1f}ms"
    )
    
    return response


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEPTION HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
# INCLUDE ROUTERS
# ═══════════════════════════════════════════════════════════════════════════════

# REST API routes
app.include_router(api_router, prefix="/api/v1")

# WebSocket routes
app.include_router(websocket_router)


# ═══════════════════════════════════════════════════════════════════════════════
# ROOT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
async def root():
    """
    Root endpoint - Basic health check.
    
    Returns service status and basic information.
    """
    return {
        "service": "MIA - Mobile İSG Aracı",
        "status": "operational",
        "version": "1.0.0",
        "model_loaded": app_state.is_model_loaded,
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "dashboard": "/dashboard",
            "docs": "/docs",
            "health": "/api/v1/health",
            "detect_video": "/api/v1/detect/video",
            "websocket": "/ws/stream"
        },
        "message": "Visit /dashboard for the Command Center UI"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Detailed health check endpoint.
    
    Returns comprehensive system health information.
    """
    return {
        "status": "healthy" if app_state.is_model_loaded else "degraded",
        "model": {
            "loaded": app_state.is_model_loaded,
            "path": settings.MODEL_PATH,
            "device": app_state.device if app_state.is_model_loaded else None
        },
        "system": {
            "debug_mode": settings.DEBUG,
            "upload_dir": settings.UPLOAD_DIR,
            "output_dir": settings.OUTPUT_DIR
        },
        "timestamp": datetime.now().isoformat()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STATIC FILES (for serving processed videos)
# ═══════════════════════════════════════════════════════════════════════════════

# Mount static files directory for output access
output_dir = Path(settings.OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(output_dir)), name="outputs")

# Mount templates directory for static assets if needed
templates_dir = Path(__file__).parent / "templates"
if templates_dir.exists():
    app.mount("/static", StaticFiles(directory=str(templates_dir)), name="static")


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD ROUTE
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/dashboard", include_in_schema=False)
async def dashboard():
    """
    Serve the Command Center Dashboard.
    
    Returns the professional SPA dashboard for monitoring.
    """
    from fastapi.responses import HTMLResponse
    
    template_path = Path(__file__).parent / "templates" / "index.html"
    
    if template_path.exists():
        return HTMLResponse(content=template_path.read_text(), status_code=200)
    else:
        return HTMLResponse(
            content="<h1>Dashboard template not found</h1><p>Place index.html in backend_api/templates/</p>",
            status_code=404
        )


@app.get("/app", include_in_schema=False)
async def app_redirect():
    """Redirect /app to /dashboard."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/dashboard")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def start_server():
    """Start the API server programmatically."""
    import uvicorn
    
    uvicorn.run(
        "backend_api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKERS,
        log_level="debug" if settings.DEBUG else "info"
    )


if __name__ == "__main__":
    start_server()
