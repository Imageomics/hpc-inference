"""
Face detection using YOLO models.
"""
from typing import List
import torch
import logging

from .base_detector import BaseDetector


class FaceDetector(BaseDetector):
    """
    YOLO-based face detector.
    
    This detector loads face detection models (e.g., from yolo-face repo)
    and performs face detection on image batches.
    """
    
    def detect(self, images: torch.Tensor, conf_threshold: float = 0.5) -> List[float]:
        """
        Detect faces in a batch of images and return detection scores.
        
        Args:
            images: Batch of preprocessed images as tensor (B, C, H, W) 
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List of detection scores (max confidence score per image)
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        # Run inference on the entire batch at once
        results = self.model(images, verbose=False)
        
        # Process detection scores
        detection_scores = []
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                confidences = result.boxes.conf
                # Keep operations on GPU until final conversion
                valid_confidences = confidences[confidences >= conf_threshold]
                if len(valid_confidences) > 0:
                    max_score = float(torch.max(valid_confidences).cpu())
                else:
                    max_score = 0.0
            else:
                max_score = 0.0
            detection_scores.append(max_score)
        
        return detection_scores
