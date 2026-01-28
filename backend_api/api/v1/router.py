"""MIA Backend - API v1 Router"""
from fastapi import APIRouter
from backend_api.api.v1.endpoints import detection, health, violations

api_router = APIRouter()
api_router.include_router(health.router, prefix="/health", tags=["Health"])
api_router.include_router(detection.router, prefix="/detect", tags=["Detection"])
api_router.include_router(violations.router, prefix="/violations", tags=["Violations"])
