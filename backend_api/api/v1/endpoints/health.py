"""MIA Backend - Health Check Endpoints"""
from datetime import datetime
from fastapi import APIRouter, Request

router = APIRouter()

@router.get("")
async def health_check(request: Request):
    app_state = request.app.state.app_state
    return {
        "status": "healthy" if app_state.is_model_loaded else "degraded",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": app_state.is_model_loaded
    }

@router.get("/detailed")
async def detailed_health(request: Request):
    app_state = request.app.state.app_state
    return {
        "status": "healthy" if app_state.is_model_loaded else "degraded",
        "timestamp": datetime.now().isoformat(),
        "model": {
            "loaded": app_state.is_model_loaded,
            "device": app_state.device if app_state.is_model_loaded else None,
        }
    }

@router.get("/ready")
async def readiness_check(request: Request):
    app_state = request.app.state.app_state
    return {"ready": app_state.is_model_loaded}

@router.get("/live")
async def liveness_check():
    return {"alive": True, "timestamp": datetime.now().isoformat()}
