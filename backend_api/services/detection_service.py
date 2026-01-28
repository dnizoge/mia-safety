"""
MIA Backend - Detection Service
================================

Service layer for detection operations.
"""

import logging
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger("mia.services.detection")


@dataclass
class Detection:
    """Single detection result."""
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    is_violation: bool = False


class DetectionService:
    """
    Service for managing detection operations.
    
    This is a thin wrapper that could be extended for:
    - Caching detection results
    - Batch processing
    - Detection filtering
    """
    
    def __init__(self):
        self.violation_classes = {
            'no-helmet', 'no_helmet', 'no helmet',
            'no-hardhat', 'no_hardhat', 'no hardhat',
            'no-vest', 'no_vest', 'no vest',
            'no-safety-vest', 'no_safety_vest',
            'no-mask', 'no_mask', 'no mask',
            'head', 'person_no_helmet'
        }
        
        self.safe_classes = {
            'helmet', 'hardhat', 'hard-hat', 'hard_hat',
            'vest', 'safety-vest', 'safety_vest', 'safety vest',
            'mask', 'safety-mask', 'safety_mask',
            'person_with_helmet', 'worker_safe'
        }
    
    def is_violation(self, class_name: str) -> bool:
        """Check if a class represents a safety violation."""
        return class_name.lower() in self.violation_classes
    
    def is_safe(self, class_name: str) -> bool:
        """Check if a class represents safe compliance."""
        return class_name.lower() in self.safe_classes
    
    def filter_violations(self, detections: List[Detection]) -> List[Detection]:
        """Filter only violation detections."""
        return [d for d in detections if d.is_violation]
    
    def filter_safe(self, detections: List[Detection]) -> List[Detection]:
        """Filter only safe/compliant detections."""
        return [d for d in detections if self.is_safe(d.class_name)]
