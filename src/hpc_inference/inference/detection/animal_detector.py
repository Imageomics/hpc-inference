"""
Animal detection using YOLO models (MegaDetector).
"""
from typing import List, Dict, Any
import torch
import numpy as np
import logging

from .base_detector import BaseDetector


class AnimalDetector(BaseDetector):
    """
    YOLO-based animal detector using MegaDetector models.
    
    This detector loads animal detection models (e.g., MegaDetector from Microsoft)
    and performs animal detection on image batches, returning both confidence scores
    and bounding box coordinates.
    """
    
    # Fallback class names (MegaDetector convention), used only when the loaded
    # model does not carry its own class-name mapping. Results normally use
    # ``result.names`` so general-purpose YOLO weights (e.g. COCO: 0=person)
    # are labeled correctly.
    CLASS_NAMES = {
        0: "animal",
        1: "person",
        2: "vehicle"
    }
    
    def detect(self, images: torch.Tensor, conf_threshold: float = 0.2) -> List[Dict[str, Any]]:
        """
        Detect animals in a batch of images and return detection results.
        
        Args:
            images: Batch of preprocessed images as tensor (B, C, H, W) 
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List[Dict[str, Any]]: List of detection results, one dict per image.
            
            Each image result dict contains:
            {
                "max_detection_score": float,      # Maximum confidence score across all detections (0.0 if none)
                "num_detections": int,             # Total number of detections above threshold
                "detections": [                    # List of individual detection objects
                    {
                        "bbox": [x1, y1, x2, y2],          # Absolute pixel coordinates (float)
                        "bbox_normalized": [x1, y1, x2, y2], # Normalized coordinates 0-1 (float)
                        "confidence": float,                 # Detection confidence score (0.0-1.0)
                        "class_id": int,                    # Numeric class ID (model-dependent; MegaDetector: 0=animal, 1=person, 2=vehicle)
                        "class_name": str                   # Human-readable class name from the model's label map
                    },
                    # ... more detections
                ]
            }
            
            Example return for batch of 2 images:
            [
                {
                    "max_detection_score": 0.85,
                    "num_detections": 2,
                    "detections": [
                        {
                            "bbox": [120.5, 80.2, 340.8, 290.1],
                            "bbox_normalized": [0.118, 0.078, 0.333, 0.284],
                            "confidence": 0.85,
                            "class_id": 0,
                            "class_name": "animal"
                        },
                        {
                            "bbox": [450.0, 200.0, 600.0, 400.0],
                            "bbox_normalized": [0.440, 0.195, 0.586, 0.391],
                            "confidence": 0.72,
                            "class_id": 1,
                            "class_name": "person"
                        }
                    ]
                },
                {
                    "max_detection_score": 0.0,
                    "num_detections": 0,
                    "detections": []
                }
            ]
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        # Run inference on the entire batch at once. The threshold MUST be passed
        # to the model: Ultralytics applies its own default cutoff (0.25) inside
        # NMS, so filtering only afterwards would silently drop detections between
        # conf_threshold and 0.25.
        results = self.model(images, conf=conf_threshold, verbose=False)
        
        # Process detection results
        batch_results = []
        for result in results:
            image_result = {
                "max_detection_score": 0.0,
                "detections": [],
                "num_detections": 0
            }
            # Class-name mapping carried by the loaded model (MegaDetector or any
            # other YOLO weights); fall back to the MegaDetector convention.
            class_names = getattr(result, "names", None) or self.CLASS_NAMES
            
            if result.boxes is not None and len(result.boxes) > 0:
                # Extract detection data
                boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                # Get image dimensions for normalization
                img_height, img_width = result.orig_shape
                
                # Filter detections by confidence threshold
                valid_indices = confidences >= conf_threshold
                
                if np.any(valid_indices):
                    valid_boxes = boxes[valid_indices]
                    valid_confidences = confidences[valid_indices]
                    valid_class_ids = class_ids[valid_indices]
                    
                    # Store maximum confidence score
                    image_result["max_detection_score"] = float(np.max(valid_confidences))
                    image_result["num_detections"] = len(valid_confidences)
                    
                    # Process each detection
                    for box, conf, class_id in zip(valid_boxes, valid_confidences, valid_class_ids):
                        x1, y1, x2, y2 = box
                        
                        detection = {
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],  # Absolute coordinates
                            "bbox_normalized": [  # Normalized coordinates [0-1]
                                float(x1 / img_width), 
                                float(y1 / img_height), 
                                float(x2 / img_width), 
                                float(y2 / img_height)
                            ],
                            "confidence": float(conf),
                            "class_id": int(class_id),
                            "class_name": class_names.get(int(class_id), "unknown")
                        }
                        image_result["detections"].append(detection)
            
            batch_results.append(image_result)
        
        return batch_results
