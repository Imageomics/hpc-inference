"""
Detection modules for various YOLO-based detection tasks.
"""

from .base_detector import BaseDetector
from .face_detector import FaceDetector
from .animal_detector import AnimalDetector

__all__ = [
    "BaseDetector",
    "FaceDetector",
    "AnimalDetector",
    "face_detect_main",
    "animal_detect_main"
]

# The pipeline entry points are re-exported lazily (PEP 562). Importing them
# eagerly here would import face_detect / animal_detect as a side effect of
# importing this package, so `python -m hpc_inference.inference.detection.animal_detect`
# would load the module twice and emit a RuntimeWarning from runpy.
_LAZY = {
    "face_detect_main": ("face_detect", "main"),
    "animal_detect_main": ("animal_detect", "main"),
}


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module
        module_name, attr = _LAZY[name]
        return getattr(import_module(f"{__name__}.{module_name}"), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
