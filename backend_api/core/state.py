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
