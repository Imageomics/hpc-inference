"""
Detection modules for various YOLO-based detection tasks.
"""

from .base_detector import BaseDetector
from .face_detector import FaceDetector
from .animal_detector import AnimalDetector
from .face_detect import main as face_detect_main
from .animal_detect import main as animal_detect_main

__all__ = [
    "BaseDetector",
    "FaceDetector",
    "AnimalDetector", 
    "face_detect_main",
    "animal_detect_main"
]