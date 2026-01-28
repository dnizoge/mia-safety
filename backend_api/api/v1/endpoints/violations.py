"""MIA Backend - Violations Endpoints"""
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from backend_api.services.violation_service import ViolationService

router = APIRouter()

class ViolationRecord(BaseModel):
    id: int
    source: str
    frame_number: Optional[int]
    timestamp_seconds: Optional[float]
    class_name: str
    confidence: float
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    created_at: str

class ViolationStats(BaseModel):
    total_violations: int
    violations_by_class: dict
    violations_last_24h: int
    violations_last_7d: int
    most_common_violation: Optional[str]

@router.get("")
async def list_violations(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    class_name: Optional[str] = None
):
    service = ViolationService()
    total, violations = await service.get_violations(page=page, page_size=page_size, class_name=class_name)
    return {"total": total, "page": page, "page_size": page_size, "violations": violations}

@router.get("/stats", response_model=ViolationStats)
async def get_violation_stats():
    service = ViolationService()
    stats = await service.get_stats()
    return ViolationStats(**stats)

@router.delete("")
async def clear_violations(confirm: bool = Query(False)):
    if not confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true to delete all violations")
    service = ViolationService()
    count = await service.clear_all_violations()
    return {"message": f"Deleted {count} violations"}
