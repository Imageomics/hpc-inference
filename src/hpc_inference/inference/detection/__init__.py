"""
Detection modules for various YOLO-based detection tasks.
"""

from .base_detector import BaseDetector
from .face_detector import FaceDetector
from .face_detect import main as face_detect_main

__all__ = [
    "BaseDetector",
    "FaceDetector",
    "face_detect_main"
]