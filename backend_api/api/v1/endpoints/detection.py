"""
MIA Backend - Detection Endpoints (Optimized for Cloud CPU)
"""

import os
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Request, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel

from backend_api.core.config import settings
from backend_api.services.violation_service import ViolationService

logger = logging.getLogger("mia.api.detection")

router = APIRouter()


class ViolationEvent(BaseModel):
    frame_number: int
    timestamp_seconds: float
    timestamp_formatted: str
    class_name: str
    confidence: float
    bbox: List[int]


class DetectionSummary(BaseModel):
    total_frames: int
    processed_frames: int
    total_violations: int
    total_safe_detections: int
    processing_time_seconds: float
    fps: float


class VideoDetectionResponse(BaseModel):
    job_id: str
    status: str
    message: str
    video_info: Optional[dict] = None
    summary: Optional[DetectionSummary] = None
    violations: Optional[List[ViolationEvent]] = None
    output_video_url: Optional[str] = None


class ImageDetectionResponse(BaseModel):
    detections: List[dict]
    violations_count: int
    safe_count: int
    processing_time_ms: float


processing_jobs = {}


@router.post("/video", response_model=VideoDetectionResponse)
async def detect_video(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence: float = 0.5,
    save_output: bool = False,
    async_processing: bool = True
):
    """Upload and process a video file for safety violations."""
    app_state = request.app.state.app_state
    
    if not app_state.is_model_loaded:
        raise HTTPException(status_code=503, detail="AI model not loaded.")
    
    if not settings.is_allowed_file(file.filename):
        raise HTTPException(status_code=400, detail=f"File type not allowed.")
    
    job_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp}_{job_id}_{file.filename}"
    upload_path = settings.get_upload_path(safe_filename)
    
    try:
        content = await file.read()
        if len(content) > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="File too large.")
        
        with open(upload_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Video saved: {upload_path} ({len(content) / (1024*1024):.1f} MB)")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")
    
    processing_jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "created_at": datetime.now().isoformat(),
        "input_file": str(upload_path),
        "result": None
    }
    
    if async_processing:
        background_tasks.add_task(
            process_video_task,
            job_id=job_id,
            video_path=str(upload_path),
            app_state=app_state,
            confidence=confidence
        )
        
        return VideoDetectionResponse(
            job_id=job_id,
            status="processing",
            message="Video uploaded. Processing in background.",
            video_info={"filename": file.filename, "size_mb": len(content) / (1024 * 1024)}
        )
    else:
        result = await process_video_task(
            job_id=job_id,
            video_path=str(upload_path),
            app_state=app_state,
            confidence=confidence
        )
        return result


@router.get("/video/{job_id}", response_model=VideoDetectionResponse)
async def get_video_detection_status(job_id: str):
    """Get the status and results of a video detection job."""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    if job["status"] == "completed" and job["result"]:
        return job["result"]
    elif job["status"] == "failed":
        return VideoDetectionResponse(
            job_id=job_id,
            status="failed",
            message=job.get("error", "Processing failed")
        )
    else:
        return VideoDetectionResponse(
            job_id=job_id,
            status=job["status"],
            message=f"Processing: {job['progress']}% complete"
        )


@router.post("/image", response_model=ImageDetectionResponse)
async def detect_image(
    request: Request,
    file: UploadFile = File(...),
    confidence: float = 0.5
):
    """Process a single image for safety violations."""
    import time
    start_time = time.time()
    
    app_state = request.app.state.app_state
    
    if not app_state.is_model_loaded:
        raise HTTPException(status_code=503, detail="AI model not loaded")
    
    try:
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")
    
    detections = await app_state.detect(image, confidence=confidence)
    
    detection_list = []
    violations_count = 0
    safe_count = 0
    
    for det in detections:
        detection_list.append({
            "class_name": det.class_name,
            "confidence": round(det.confidence, 3),
            "bbox": list(det.bbox),
            "is_violation": det.is_violation
        })
        
        if det.is_violation:
            violations_count += 1
        elif app_state._is_safe_class(det.class_name):
            safe_count += 1
    
    processing_time = (time.time() - start_time) * 1000
    
    return ImageDetectionResponse(
        detections=detection_list,
        violations_count=violations_count,
        safe_count=safe_count,
        processing_time_ms=round(processing_time, 2)
    )


async def process_video_task(
    job_id: str,
    video_path: str,
    app_state,
    confidence: float = 0.5
) -> VideoDetectionResponse:
    """
    Fast video processing - samples frames, no output video.
    Optimized for cloud CPU deployment.
    """
    import time
    start_time = time.time()
    
    processing_jobs[job_id]["status"] = "processing"
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception("Failed to open video file")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Processing video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames, {duration:.1f}s")
        
        # AGGRESSIVE FRAME SAMPLING for speed
        # Process only ~2 frames per second max
        target_samples_per_second = 2
        frame_interval = max(1, int(fps / target_samples_per_second))
        
        # Limit total frames to process (max 50 samples for demo)
        max_samples = 50
        total_samples = min(max_samples, total_frames // frame_interval)
        
        logger.info(f"Sampling every {frame_interval} frames, ~{total_samples} samples total")
        
        violations: List[ViolationEvent] = []
        frames_analyzed = 0
        total_violations = 0
        total_safe = 0
        
        violation_service = ViolationService()
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Update progress
            progress = int((frame_count / total_frames) * 100)
            processing_jobs[job_id]["progress"] = progress
            
            # Only process sampled frames
            if frame_count % frame_interval != 0:
                continue
            
            # Stop if we've analyzed enough
            if frames_analyzed >= max_samples:
                break
            
            frames_analyzed += 1
            
            # Run detection
            detections = await app_state.detect(frame, confidence=confidence)
            
            for det in detections:
                if det.is_violation:
                    total_violations += 1
                    
                    timestamp_sec = frame_count / fps
                    timestamp_fmt = f"{int(timestamp_sec // 60):02d}:{timestamp_sec % 60:05.2f}"
                    
                    violation_event = ViolationEvent(
                        frame_number=frame_count,
                        timestamp_seconds=round(timestamp_sec, 2),
                        timestamp_formatted=timestamp_fmt,
                        class_name=det.class_name,
                        confidence=round(det.confidence, 3),
                        bbox=list(det.bbox)
                    )
                    violations.append(violation_event)
                    
                    await violation_service.log_violation(
                        source=video_path,
                        frame_number=frame_count,
                        timestamp=timestamp_sec,
                        class_name=det.class_name,
                        confidence=det.confidence,
                        bbox=det.bbox
                    )
                
                elif app_state._is_safe_class(det.class_name):
                    total_safe += 1
            
            if frames_analyzed % 10 == 0:
                logger.info(f"Job {job_id}: analyzed {frames_analyzed}/{total_samples} samples")
        
        cap.release()
        
        processing_time = time.time() - start_time
        actual_fps = frames_analyzed / processing_time if processing_time > 0 else 0
        
        result = VideoDetectionResponse(
            job_id=job_id,
            status="completed",
            message=f"Analyzed {frames_analyzed} frames in {processing_time:.1f}s",
            video_info={
                "width": width,
                "height": height,
                "fps": fps,
                "duration_seconds": round(duration, 2)
            },
            summary=DetectionSummary(
                total_frames=total_frames,
                processed_frames=frames_analyzed,
                total_violations=total_violations,
                total_safe_detections=total_safe,
                processing_time_seconds=round(processing_time, 2),
                fps=round(actual_fps, 2)
            ),
            violations=violations,
            output_video_url=None
        )
        
        processing_jobs[job_id]["status"] = "completed"
        processing_jobs[job_id]["result"] = result
        
        logger.info(f"Job {job_id} completed: {total_violations} violations, {total_safe} safe in {processing_time:.1f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        import traceback
        traceback.print_exc()
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["error"] = str(e)
        
        return VideoDetectionResponse(
            job_id=job_id,
            status="failed",
            message=f"Processing failed: {str(e)}"
        )
