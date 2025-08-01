"""
Base detector class for YOLO-based detection models.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import logging

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None


def check_yolo_dependencies() -> None:
    """Check if YOLO dependencies are available."""
    if not YOLO_AVAILABLE:
        raise ImportError(
            "Ultralytics YOLO is not installed. Install with: "
            "pip install 'hpc-inference[yolo]' or pip install ultralytics"
        )


class BaseDetector(ABC):
    """
    Base class for YOLO-based detection models.
    
    This class provides a common interface for different detection tasks
    (face detection, animal detection, etc.) that use YOLO models.
    """
    
    def __init__(self, config: Dict[str, Any], device: Optional[str] = None):
        """
        Initialize the detector.
        
        Args:
            config: Configuration dictionary containing model parameters
            device: Device to run the model on (e.g., 'cuda:0', 'cpu'). 
                   If None, will auto-detect.
        """
        check_yolo_dependencies()
        
        self.config = config
        self.model = None
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self) -> None:
        """Load the YOLO model from weights."""
        if "model" not in self.config:
            raise ValueError("Config must contain 'model' section")
        
        model_config = self.config["model"]
        if "weights" not in model_config:
            raise ValueError("Model config must contain 'weights' field")
            
        weights_path = model_config["weights"]
        logging.info(f"Loading YOLO model from: {weights_path}")
        
        self.model = YOLO(weights_path)
        self.model.to(self.device)
        
        # Note: YOLO models are automatically in eval mode for inference
        # Calling .eval() can trigger training initialization, so we avoid it
        
        logging.info(f"Model loaded successfully on device: {self.device}")
        
    @abstractmethod
    def detect(self, images: torch.Tensor, conf_threshold: float = 0.5) -> Any:
        """
        Perform detection on a batch of images.
        
        Args:
            images: Batch of preprocessed images as tensor (B, C, H, W)
            conf_threshold: Confidence threshold for detections
            
        Returns:
            Detection results (format depends on specific detector implementation)
        """
        pass
        
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.model is not None
