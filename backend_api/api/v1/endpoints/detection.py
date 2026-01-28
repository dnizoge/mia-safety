"""
MIA Backend - Detection Endpoints
==================================

Endpoints for video upload and detection processing.
"""

import os
import uuid
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Request, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend_api.core.config import settings
from backend_api.services.detection_service import DetectionService
from backend_api.services.violation_service import ViolationService

logger = logging.getLogger("mia.api.detection")

router = APIRouter()


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class ViolationEvent(BaseModel):
    """Single violation event in the video."""
    frame_number: int
    timestamp_seconds: float
    timestamp_formatted: str
    class_name: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]


class DetectionSummary(BaseModel):
    """Summary of detections in a video."""
    total_frames: int
    processed_frames: int
    total_violations: int
    total_safe_detections: int
    processing_time_seconds: float
    fps: float


class VideoDetectionResponse(BaseModel):
    """Response for video detection endpoint."""
    job_id: str
    status: str
    message: str
    video_info: Optional[dict] = None
    summary: Optional[DetectionSummary] = None
    violations: Optional[List[ViolationEvent]] = None
    output_video_url: Optional[str] = None


class ImageDetectionResponse(BaseModel):
    """Response for single image detection."""
    detections: List[dict]
    violations_count: int
    safe_count: int
    processing_time_ms: float


# ═══════════════════════════════════════════════════════════════════════════════
# JOB TRACKING (In-memory for simplicity)
# ═══════════════════════════════════════════════════════════════════════════════

# Simple in-memory job tracking (use Redis in production)
processing_jobs = {}


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/video", response_model=VideoDetectionResponse)
async def detect_video(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence: float = 0.5,
    save_output: bool = True,
    async_processing: bool = True
):
    """
    Upload and process a video file for safety violations.
    
    This endpoint accepts a video file, processes it frame by frame,
    and returns a detailed report of all detected violations with timestamps.
    
    Args:
        file: Video file (mp4, avi, mov, mkv, webm)
        confidence: Detection confidence threshold (0.0 - 1.0)
        save_output: Whether to save annotated output video
        async_processing: Process asynchronously (returns job_id immediately)
        
    Returns:
        Job ID and status (if async) or full results (if sync)
    """
    app_state = request.app.state.app_state
    
    # Validate model is loaded
    if not app_state.is_model_loaded:
        raise HTTPException(
            status_code=503,
            detail="AI model not loaded. Server is starting up."
        )
    
    # Validate file extension
    if not settings.is_allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Supported: {settings.ALLOWED_EXTENSIONS}"
        )
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save uploaded file
    safe_filename = f"{timestamp}_{job_id}_{file.filename}"
    upload_path = settings.get_upload_path(safe_filename)
    
    try:
        # Save file to disk
        content = await file.read()
        
        # Check file size
        if len(content) > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE // (1024*1024)} MB"
            )
        
        with open(upload_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Video saved: {upload_path} ({len(content) / (1024*1024):.1f} MB)")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")
    
    # Initialize job status
    processing_jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "created_at": datetime.now().isoformat(),
        "input_file": str(upload_path),
        "output_file": None,
        "result": None
    }
    
    if async_processing:
        # Process in background
        background_tasks.add_task(
            process_video_task,
            job_id=job_id,
            video_path=str(upload_path),
            app_state=app_state,
            confidence=confidence,
            save_output=save_output
        )
        
        return VideoDetectionResponse(
            job_id=job_id,
            status="processing",
            message="Video uploaded successfully. Processing in background.",
            video_info={
                "filename": file.filename,
                "size_mb": len(content) / (1024 * 1024)
            }
        )
    
    else:
        # Process synchronously
        result = await process_video_task(
            job_id=job_id,
            video_path=str(upload_path),
            app_state=app_state,
            confidence=confidence,
            save_output=save_output
        )
        
        return result


@router.get("/video/{job_id}", response_model=VideoDetectionResponse)
async def get_video_detection_status(job_id: str):
    """
    Get the status and results of a video detection job.
    
    Args:
        job_id: Job ID returned from POST /detect/video
        
    Returns:
        Job status and results (if complete)
    """
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
    """
    Process a single image for safety violations.
    
    Args:
        file: Image file (jpg, png, bmp, webp)
        confidence: Detection confidence threshold
        
    Returns:
        List of detections with violation flags
    """
    import time
    start_time = time.time()
    
    app_state = request.app.state.app_state
    
    # Validate model is loaded
    if not app_state.is_model_loaded:
        raise HTTPException(
            status_code=503,
            detail="AI model not loaded"
        )
    
    # Read image
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
    
    # Run detection
    detections = await app_state.detect(image, confidence=confidence)
    
    # Format results
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


# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND TASK
# ═══════════════════════════════════════════════════════════════════════════════

async def process_video_task(
    job_id: str,
    video_path: str,
    app_state,
    confidence: float = 0.5,
    save_output: bool = True
) -> VideoDetectionResponse:
    """
    Background task for video processing.
    
    Processes video frame by frame, detects violations,
    and generates a detailed report.
    """
    import time
    start_time = time.time()
    
    # Update job status
    processing_jobs[job_id]["status"] = "processing"
    
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception("Failed to open video file")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
        
        # Initialize output writer
        writer = None
        output_path = None
        
        if save_output:
            output_filename = f"processed_{job_id}.mp4"
            output_path = settings.get_output_path(output_filename)
            
            # Use H.264 codec for browser compatibility
            # Try avc1 first (macOS), fallback to mp4v
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (width, height)
            )
            
            # Fallback if avc1 doesn't work
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    fps,
                    (width, height)
                )
        
        # Process frames
        violations: List[ViolationEvent] = []
        frame_count = 0
        total_violations = 0
        total_safe = 0
        
        # Violation logging service
        violation_service = ViolationService()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Update progress
            progress = int((frame_count / total_frames) * 100)
            processing_jobs[job_id]["progress"] = progress
            
            # Run detection
            detections = await app_state.detect(frame, confidence=confidence)
            
            # Process detections
            for det in detections:
                if det.is_violation:
                    total_violations += 1
                    
                    # Calculate timestamp
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
                    
                    # Log to database
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
            
            # Draw on frame and write to output
            if writer is not None:
                annotated = draw_detections(frame, detections)
                writer.write(annotated)
            
            # Log progress periodically
            if frame_count % 100 == 0:
                logger.info(f"Job {job_id}: {progress}% ({frame_count}/{total_frames})")
        
        # Cleanup
        cap.release()
        if writer is not None:
            writer.release()
        
        # Convert to browser-compatible format using ffmpeg
        if output_path and output_path.exists():
            import subprocess
            final_output = output_path.parent / f"web_{job_id}.mp4"
            try:
                subprocess.run([
                    'ffmpeg', '-y', '-i', str(output_path),
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                    '-movflags', '+faststart',  # Enable streaming
                    '-pix_fmt', 'yuv420p',  # Browser compatible pixel format
                    str(final_output)
                ], check=True, capture_output=True, timeout=300)
                
                # Remove the original and use the web version
                output_path.unlink()
                output_path = final_output
                logger.info(f"Converted to web-compatible format: {final_output}")
            except Exception as conv_err:
                logger.warning(f"FFmpeg conversion failed, using original: {conv_err}")
        
        # Calculate processing stats
        processing_time = time.time() - start_time
        actual_fps = frame_count / processing_time if processing_time > 0 else 0
        
        # Build response
        result = VideoDetectionResponse(
            job_id=job_id,
            status="completed",
            message=f"Processed {frame_count} frames in {processing_time:.1f}s",
            video_info={
                "width": width,
                "height": height,
                "fps": fps,
                "duration_seconds": total_frames / fps if fps > 0 else 0
            },
            summary=DetectionSummary(
                total_frames=total_frames,
                processed_frames=frame_count,
                total_violations=total_violations,
                total_safe_detections=total_safe,
                processing_time_seconds=round(processing_time, 2),
                fps=round(actual_fps, 2)
            ),
            violations=violations,
            output_video_url=f"/outputs/{output_path.name}" if output_path else None
        )
        
        # Update job status
        processing_jobs[job_id]["status"] = "completed"
        processing_jobs[job_id]["result"] = result
        processing_jobs[job_id]["output_file"] = str(output_path) if output_path else None
        
        logger.info(f"Job {job_id} completed: {total_violations} violations found")
        
        return result
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["error"] = str(e)
        
        return VideoDetectionResponse(
            job_id=job_id,
            status="failed",
            message=f"Processing failed: {str(e)}"
        )


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw detection boxes on frame."""
    annotated = frame.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        
        # Color based on violation status
        if det.is_violation:
            color = (0, 0, 255)  # RED
            label = f"VIOLATION: {det.class_name}"
        else:
            color = (0, 255, 0)  # GREEN
            label = det.class_name
        
        label += f" {det.confidence:.2f}"
        
        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        cv2.rectangle(
            annotated,
            (x1, y1 - label_h - 10),
            (x1 + label_w + 5, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            annotated,
            label,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
    
    return annotated
