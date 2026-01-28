#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# MIA - Fix Missing Files Script
# ═══════════════════════════════════════════════════════════════════════════════
# Run this from inside the mia_project folder to create missing files

echo "🔧 Creating missing backend files..."

# Create the router.py file
cat > backend_api/api/v1/router.py << 'ROUTER_EOF'
"""MIA Backend - API v1 Router"""
from fastapi import APIRouter
from backend_api.api.v1.endpoints import detection, health, violations

api_router = APIRouter()
api_router.include_router(health.router, prefix="/health", tags=["Health"])
api_router.include_router(detection.router, prefix="/detect", tags=["Detection"])
api_router.include_router(violations.router, prefix="/violations", tags=["Violations"])
ROUTER_EOF
echo "   ✓ Created router.py"

# Create health.py
cat > backend_api/api/v1/endpoints/health.py << 'HEALTH_EOF'
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
HEALTH_EOF
echo "   ✓ Created health.py"

# Create violations.py
cat > backend_api/api/v1/endpoints/violations.py << 'VIOLATIONS_EOF'
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
VIOLATIONS_EOF
echo "   ✓ Created violations.py"

# Create __init__.py files
echo '"""MIA Backend API v1 Endpoints"""' > backend_api/api/v1/endpoints/__init__.py
echo '"""MIA Backend API v1"""' > backend_api/api/v1/__init__.py
echo '"""MIA Backend API"""' > backend_api/api/__init__.py
echo '"""MIA Backend WebSocket"""' > backend_api/api/websocket/__init__.py
echo '"""MIA Backend Core"""' > backend_api/core/__init__.py
echo '"""MIA Backend DB"""' > backend_api/db/__init__.py
echo '"""MIA Backend Services"""' > backend_api/services/__init__.py
echo '"""MIA Backend"""' > backend_api/__init__.py
echo "   ✓ Created __init__.py files"

# Create state.py if missing
if [ ! -f backend_api/core/state.py ]; then
cat > backend_api/core/state.py << 'STATE_EOF'
"""MIA Backend - Application State"""
import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import numpy as np

try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None
    torch = None

logger = logging.getLogger("mia.api.state")

@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    class_id: int
    class_name: str
    confidence: float
    is_violation: bool

class AppState:
    VIOLATION_CLASSES = {"no_helmet", "no_vest", "no_harness", "no-helmet", "no-vest", "no-harness", "head"}
    SAFE_CLASSES = {"helmet", "vest", "harness", "hard-hat", "hardhat", "safety-vest"}
    
    def __init__(self):
        self._model = None
        self._device = "cpu"
        self._class_names = {}
        self._lock = asyncio.Lock()
        self._is_loaded = False
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
    
    @property
    def is_model_loaded(self) -> bool:
        return self._is_loaded and self._model is not None
    
    @property
    def device(self) -> str:
        return self._device
    
    @property
    def class_names(self) -> Dict[int, str]:
        return self._class_names
    
    async def load_model(self, model_path: str, device: str = "auto") -> bool:
        async with self._lock:
            if self._is_loaded:
                return True
            if not YOLO_AVAILABLE:
                logger.error("YOLO not available!")
                return False
            if not Path(model_path).exists():
                logger.error(f"Model not found: {model_path}")
                return False
            try:
                self._device = self._detect_device(device)
                loop = asyncio.get_event_loop()
                self._model = await loop.run_in_executor(None, lambda: YOLO(model_path))
                if hasattr(self._model, 'names'):
                    self._class_names = self._model.names
                self._is_loaded = True
                logger.info(f"Model loaded on {self._device}")
                return True
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return False
    
    async def unload_model(self):
        async with self._lock:
            self._model = None
            self._is_loaded = False
    
    def _detect_device(self, preferred: str = "auto") -> str:
        if preferred != "auto":
            return preferred
        if torch is None:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _is_violation_class(self, class_name: str) -> bool:
        return class_name.lower().strip() in self.VIOLATION_CLASSES
    
    def _is_safe_class(self, class_name: str) -> bool:
        return class_name.lower().strip() in self.SAFE_CLASSES
    
    async def detect(self, frame: np.ndarray, confidence: Optional[float] = None) -> List[Detection]:
        if not self.is_model_loaded:
            return []
        conf = confidence or self.confidence_threshold
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._model.predict(source=frame, conf=conf, iou=self.iou_threshold, device=self._device, verbose=False)
            )
            detections = []
            for result in results:
                if result.boxes is None:
                    continue
                for i in range(len(result.boxes)):
                    xyxy = result.boxes.xyxy[i].cpu().numpy().astype(int)
                    class_id = int(result.boxes.cls[i].cpu().numpy())
                    conf_score = float(result.boxes.conf[i].cpu().numpy())
                    class_name = self._class_names.get(class_id, f"class_{class_id}")
                    detections.append(Detection(
                        bbox=tuple(xyxy),
                        class_id=class_id,
                        class_name=class_name,
                        confidence=conf_score,
                        is_violation=self._is_violation_class(class_name)
                    ))
            return detections
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
STATE_EOF
echo "   ✓ Created state.py"
fi

# Create config.py if missing
if [ ! -f backend_api/core/config.py ]; then
cat > backend_api/core/config.py << 'CONFIG_EOF'
"""MIA Backend - Configuration"""
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    APP_NAME: str = "MIA"
    DEBUG: bool = Field(default=True, alias="MIA_DEBUG")
    HOST: str = Field(default="0.0.0.0", alias="MIA_HOST")
    PORT: int = Field(default=8000, alias="MIA_PORT")
    WORKERS: int = 4
    MODEL_PATH: str = Field(default="weights/best_mia.pt", alias="MIA_MODEL_PATH")
    CONFIDENCE_THRESHOLD: float = 0.5
    DATABASE_URL: str = Field(default="sqlite+aiosqlite:///./data/mia_violations.db", alias="MIA_DATABASE_URL")
    UPLOAD_DIR: str = "data/uploads"
    OUTPUT_DIR: str = "data/outputs"
    MAX_UPLOAD_SIZE: int = 500 * 1024 * 1024
    ALLOWED_EXTENSIONS: List[str] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    LOG_DIR: str = "logs/api"
    CORS_ORIGINS: List[str] = ["*"]
    WS_MAX_CONNECTIONS: int = 10
    
    class Config:
        env_prefix = "MIA_"
        env_file = ".env"
    
    def get_upload_path(self, filename: str) -> Path:
        Path(self.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        return Path(self.UPLOAD_DIR) / filename
    
    def get_output_path(self, filename: str) -> Path:
        Path(self.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        return Path(self.OUTPUT_DIR) / filename
    
    def is_allowed_file(self, filename: str) -> bool:
        return Path(filename).suffix.lower() in self.ALLOWED_EXTENSIONS

settings = Settings()
CONFIG_EOF
echo "   ✓ Created config.py"
fi

# Create session.py if missing
if [ ! -f backend_api/db/session.py ]; then
cat > backend_api/db/session.py << 'SESSION_EOF'
"""MIA Backend - Database Session"""
import logging
from pathlib import Path
from typing import Optional
import aiosqlite
from backend_api.core.config import settings

logger = logging.getLogger("mia.api.db")
_db_connection: Optional[aiosqlite.Connection] = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    frame_number INTEGER,
    timestamp_seconds REAL,
    class_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    bbox_x1 INTEGER NOT NULL,
    bbox_y1 INTEGER NOT NULL,
    bbox_x2 INTEGER NOT NULL,
    bbox_y2 INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_violations_class ON violations(class_name);
CREATE INDEX IF NOT EXISTS idx_violations_created ON violations(created_at);
"""

async def init_db():
    global _db_connection
    db_url = settings.DATABASE_URL
    db_path = db_url.replace("sqlite+aiosqlite:///", "").replace("sqlite:///", "")
    if db_path.startswith("./"):
        db_path = db_path[2:]
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    _db_connection = await aiosqlite.connect(db_path)
    await _db_connection.execute("PRAGMA journal_mode=WAL")
    await _db_connection.executescript(SCHEMA)
    await _db_connection.commit()
    logger.info(f"Database initialized: {db_path}")

async def close_db():
    global _db_connection
    if _db_connection:
        await _db_connection.close()
        _db_connection = None

async def get_db() -> aiosqlite.Connection:
    if _db_connection is None:
        raise RuntimeError("Database not initialized")
    return _db_connection
SESSION_EOF
echo "   ✓ Created session.py"
fi

# Create violation_service.py if missing
if [ ! -f backend_api/services/violation_service.py ]; then
cat > backend_api/services/violation_service.py << 'SERVICE_EOF'
"""MIA Backend - Violation Service"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from backend_api.db.session import get_db

logger = logging.getLogger("mia.api.violation_service")

class ViolationService:
    async def log_violation(self, source: str, class_name: str, confidence: float, bbox: Tuple[int, int, int, int], frame_number: Optional[int] = None, timestamp: Optional[float] = None) -> int:
        try:
            db = await get_db()
            x1, y1, x2, y2 = bbox
            cursor = await db.execute(
                "INSERT INTO violations (source, frame_number, timestamp_seconds, class_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (source, frame_number, timestamp, class_name, confidence, x1, y1, x2, y2)
            )
            await db.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to log violation: {e}")
            return -1
    
    async def get_violations(self, page: int = 1, page_size: int = 50, class_name: Optional[str] = None, source: Optional[str] = None, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Tuple[int, List[Dict]]:
        try:
            db = await get_db()
            conditions, params = [], []
            if class_name:
                conditions.append("class_name = ?")
                params.append(class_name)
            where = " AND ".join(conditions) if conditions else "1=1"
            cursor = await db.execute(f"SELECT COUNT(*) FROM violations WHERE {where}", params)
            total = (await cursor.fetchone())[0]
            offset = (page - 1) * page_size
            cursor = await db.execute(f"SELECT * FROM violations WHERE {where} ORDER BY created_at DESC LIMIT ? OFFSET ?", params + [page_size, offset])
            rows = await cursor.fetchall()
            violations = [{"id": r[0], "source": r[1], "frame_number": r[2], "timestamp_seconds": r[3], "class_name": r[4], "confidence": r[5], "bbox_x1": r[6], "bbox_y1": r[7], "bbox_x2": r[8], "bbox_y2": r[9], "created_at": r[10]} for r in rows]
            return total, violations
        except Exception as e:
            logger.error(f"Failed to get violations: {e}")
            return 0, []
    
    async def get_stats(self) -> Dict[str, Any]:
        try:
            db = await get_db()
            cursor = await db.execute("SELECT COUNT(*) FROM violations")
            total = (await cursor.fetchone())[0]
            cursor = await db.execute("SELECT class_name, COUNT(*) FROM violations GROUP BY class_name ORDER BY COUNT(*) DESC")
            by_class = {r[0]: r[1] for r in await cursor.fetchall()}
            yesterday = (datetime.now() - timedelta(days=1)).isoformat()
            cursor = await db.execute("SELECT COUNT(*) FROM violations WHERE created_at >= ?", (yesterday,))
            last_24h = (await cursor.fetchone())[0]
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            cursor = await db.execute("SELECT COUNT(*) FROM violations WHERE created_at >= ?", (week_ago,))
            last_7d = (await cursor.fetchone())[0]
            return {"total_violations": total, "violations_by_class": by_class, "violations_last_24h": last_24h, "violations_last_7d": last_7d, "most_common_violation": list(by_class.keys())[0] if by_class else None}
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"total_violations": 0, "violations_by_class": {}, "violations_last_24h": 0, "violations_last_7d": 0, "most_common_violation": None}
    
    async def clear_all_violations(self) -> int:
        try:
            db = await get_db()
            cursor = await db.execute("DELETE FROM violations")
            await db.commit()
            return cursor.rowcount
        except:
            return 0
SERVICE_EOF
echo "   ✓ Created violation_service.py"
fi

# Create stream_handler.py if missing
if [ ! -f backend_api/api/websocket/stream_handler.py ]; then
cat > backend_api/api/websocket/stream_handler.py << 'STREAM_EOF'
"""MIA Backend - WebSocket Stream Handler"""
from fastapi import APIRouter
router = APIRouter()

@router.get("/ws/status")
async def websocket_status():
    return {"status": "WebSocket endpoint available at /ws/stream"}
STREAM_EOF
echo "   ✓ Created stream_handler.py (minimal)"
fi

echo ""
echo "✅ All missing files created!"
echo ""
echo "Now run:"
echo "   uvicorn backend_api.main:app --reload --host 0.0.0.0 --port 8000"
