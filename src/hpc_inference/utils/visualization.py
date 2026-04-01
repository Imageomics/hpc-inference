"""
Visualization utilities for detection results.

This module provides functions to visualize detection results on original images,
handling the coordinate transformation from preprocessed (letterbox) coordinates
back to original image coordinates.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors as mcolors
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Tuple, Union, Optional
import torch


def reverse_letterbox_coords(
    bbox: List[float], 
    original_shape: Tuple[int, int], 
    letterbox_shape: Tuple[int, int]
) -> List[float]:
    """
    Convert bounding box coordinates from letterbox (preprocessed) space back to original image space.
    
    The letterbox transform preserves aspect ratio by:
    1. Scaling the image to fit within the target size
    2. Adding symmetric padding to make it square
    
    This function reverses that transformation to map coordinates back to the original image.
    
    Args:
        bbox (List[float]): Bounding box coordinates in letterbox space [x1, y1, x2, y2]
        original_shape (Tuple[int, int]): Original image dimensions (height, width)
        letterbox_shape (Tuple[int, int]): Letterbox dimensions (height, width), typically (1280, 1280)
        
    Returns:
        List[float]: Bounding box coordinates in original image space [x1, y1, x2, y2]
        
    Example:
        >>> # Original image is 1200x800, letterbox is 1280x1280
        >>> # Detection bbox in letterbox space
        >>> letterbox_bbox = [640.0, 320.0, 960.0, 640.0]
        >>> original_bbox = reverse_letterbox_coords(
        ...     letterbox_bbox, 
        ...     (800, 1200),  # original (H, W)
        ...     (1280, 1280)  # letterbox (H, W)
        ... )
        >>> print(original_bbox)  # Coordinates in original 1200x800 image
    """
    x1, y1, x2, y2 = bbox
    orig_h, orig_w = original_shape
    letterbox_h, letterbox_w = letterbox_shape
    
    # Calculate the scaling ratio used in letterbox transform
    # The letterbox transform uses min ratio to preserve aspect ratio
    scale_ratio = min(letterbox_h / orig_h, letterbox_w / orig_w)
    
    # Calculate the scaled dimensions (before padding)
    scaled_w = int(round(orig_w * scale_ratio))
    scaled_h = int(round(orig_h * scale_ratio))
    
    # Calculate the padding added
    pad_w = (letterbox_w - scaled_w) / 2
    pad_h = (letterbox_h - scaled_h) / 2
    
    # Remove padding from coordinates
    x1_unpadded = x1 - pad_w
    y1_unpadded = y1 - pad_h
    x2_unpadded = x2 - pad_w
    y2_unpadded = y2 - pad_h
    
    # Scale back to original image size
    x1_original = x1_unpadded / scale_ratio
    y1_original = y1_unpadded / scale_ratio
    x2_original = x2_unpadded / scale_ratio
    y2_original = y2_unpadded / scale_ratio
    
    # Ensure coordinates are within image bounds
    x1_original = max(0, min(x1_original, orig_w))
    y1_original = max(0, min(y1_original, orig_h))
    x2_original = max(0, min(x2_original, orig_w))
    y2_original = max(0, min(y2_original, orig_h))
    
    return [x1_original, y1_original, x2_original, y2_original]


def get_class_colors() -> Dict[str, str]:
    """
    Get predefined colors for different detection classes.
    
    Returns:
        Dict[str, str]: Mapping of class names to hex color codes
    """
    return {
        'animal': '#FF6B6B',    # Red
        'person': '#4ECDC4',    # Teal
        'vehicle': '#45B7D1',   # Blue
        'unknown': '#96CEB4'    # Green
    }


def plot_detections_matplotlib(
    image: Union[np.ndarray, Image.Image], 
    detections: List[Dict[str, Any]],
    original_shape: Optional[Tuple[int, int]] = None,
    letterbox_shape: Tuple[int, int] = (1280, 1280),
    confidence_threshold: float = 0.0,
    show_confidence: bool = True,
    show_class_names: bool = True,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Detection Results"
) -> plt.Figure:
    """
    Plot detection results on an image using matplotlib.
    
    Args:
        image (np.ndarray or PIL.Image): Original image to plot detections on
        detections (List[Dict]): List of detection dictionaries from AnimalDetector.detect()
        original_shape (Tuple[int, int], optional): Original image shape (height, width).
            If None, inferred from image.
        letterbox_shape (Tuple[int, int]): Shape used during preprocessing (height, width)
        confidence_threshold (float): Only show detections above this threshold
        show_confidence (bool): Whether to show confidence scores in labels
        show_class_names (bool): Whether to show class names in labels
        figsize (Tuple[int, int]): Figure size for matplotlib
        title (str): Plot title
        
    Returns:
        plt.Figure: Matplotlib figure with plotted detections
        
    Example:
        >>> from PIL import Image
        >>> # Load original image and detection results
        >>> img = Image.open("wildlife_photo.jpg")
        >>> detections = [
        ...     {
        ...         "bbox": [640.0, 320.0, 960.0, 640.0],  # letterbox coordinates
        ...         "confidence": 0.85,
        ...         "class_name": "animal"
        ...     }
        ... ]
        >>> fig = plot_detections_matplotlib(img, detections, show_confidence=True)
        >>> plt.show()
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image_array = np.array(image)
        if original_shape is None:
            original_shape = (image.height, image.width)
    else:
        image_array = image
        if original_shape is None:
            original_shape = (image.shape[0], image.shape[1])
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image_array)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Get class colors
    class_colors = get_class_colors()
    
    # Plot each detection
    detection_count = 0
    for detection in detections:
        confidence = detection.get('confidence', 0.0)
        
        # Skip low confidence detections
        if confidence < confidence_threshold:
            continue
            
        # Get detection info
        bbox_letterbox = detection['bbox']
        class_name = detection.get('class_name', 'unknown')
        
        # Convert coordinates back to original image space
        bbox_original = reverse_letterbox_coords(bbox_letterbox, original_shape, letterbox_shape)
        x1, y1, x2, y2 = bbox_original
        
        # Calculate box dimensions
        width = x2 - x1
        height = y2 - y1
        
        # Skip very small or invalid boxes
        if width <= 0 or height <= 0:
            continue
            
        # Get color for this class
        color = class_colors.get(class_name, class_colors['unknown'])
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x1, y1), width, height, 
            linewidth=2, 
            edgecolor=color, 
            facecolor='none',
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # Create label
        label_parts = []
        if show_class_names:
            label_parts.append(class_name)
        if show_confidence:
            label_parts.append(f"{confidence:.2f}")
        
        if label_parts:
            label = " | ".join(label_parts)
            
            # Add text background for better readability
            ax.text(
                x1, y1 - 5, label,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                fontsize=10, color='white', weight='bold'
            )
        
        detection_count += 1
    
    # Add summary text
    ax.text(
        0.02, 0.98, f"Detections found: {detection_count}", 
        transform=ax.transAxes, 
        fontsize=12, 
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7),
        color='white', weight='bold'
    )
    
    plt.tight_layout()
    return fig


def plot_detections_pil(
    image: Union[np.ndarray, Image.Image], 
    detections: List[Dict[str, Any]],
    original_shape: Optional[Tuple[int, int]] = None,
    letterbox_shape: Tuple[int, int] = (1280, 1280),
    confidence_threshold: float = 0.0,
    show_confidence: bool = True,
    show_class_names: bool = True,
    box_width: int = 3
) -> Image.Image:
    """
    Plot detection results on an image using PIL (faster, no matplotlib dependency).
    
    Args:
        image (np.ndarray or PIL.Image): Original image to plot detections on
        detections (List[Dict]): List of detection dictionaries from AnimalDetector.detect()
        original_shape (Tuple[int, int], optional): Original image shape (height, width).
            If None, inferred from image.
        letterbox_shape (Tuple[int, int]): Shape used during preprocessing (height, width)
        confidence_threshold (float): Only show detections above this threshold
        show_confidence (bool): Whether to show confidence scores in labels
        show_class_names (bool): Whether to show class names in labels
        box_width (int): Width of bounding box lines
        
    Returns:
        PIL.Image: Image with detection boxes drawn
        
    Example:
        >>> from PIL import Image
        >>> # Load original image and detection results
        >>> img = Image.open("wildlife_photo.jpg")
        >>> detections = [...]  # From AnimalDetector.detect()
        >>> result_img = plot_detections_pil(img, detections)
        >>> result_img.save("detection_results.jpg")
    """
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        if original_shape is None:
            original_shape = (image.shape[0], image.shape[1])
    else:
        if original_shape is None:
            original_shape = (image.height, image.width)
    
    # Create a copy to draw on
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Color mapping (RGB tuples)
    class_colors = {
        'animal': (255, 107, 107),    # Red
        'person': (78, 205, 196),     # Teal
        'vehicle': (69, 183, 209),    # Blue
        'unknown': (150, 206, 180)    # Green
    }
    
    # Plot each detection
    detection_count = 0
    for detection in detections:
        confidence = detection.get('confidence', 0.0)
        
        # Skip low confidence detections
        if confidence < confidence_threshold:
            continue
            
        # Get detection info
        bbox_letterbox = detection['bbox']
        class_name = detection.get('class_name', 'unknown')
        
        # Convert coordinates back to original image space
        bbox_original = reverse_letterbox_coords(bbox_letterbox, original_shape, letterbox_shape)
        x1, y1, x2, y2 = bbox_original
        
        # Skip very small or invalid boxes
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
            
        # Get color for this class
        color = class_colors.get(class_name, class_colors['unknown'])
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)
        
        # Create label
        label_parts = []
        if show_class_names:
            label_parts.append(class_name)
        if show_confidence:
            label_parts.append(f"{confidence:.2f}")
        
        if label_parts:
            label = " | ".join(label_parts)
            
            # Calculate text size and position
            if font:
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width = len(label) * 8  # Rough estimation
                text_height = 14
            
            # Draw text background
            text_x = x1
            text_y = max(0, y1 - text_height - 5)
            draw.rectangle(
                [text_x, text_y, text_x + text_width + 6, text_y + text_height + 4], 
                fill=color
            )
            
            # Draw text
            draw.text(
                (text_x + 3, text_y + 2), label, 
                fill='white', font=font
            )
        
        detection_count += 1
    
    return result_image


def save_detection_visualization(
    image: Union[np.ndarray, Image.Image], 
    detections: List[Dict[str, Any]],
    output_path: str,
    original_shape: Optional[Tuple[int, int]] = None,
    letterbox_shape: Tuple[int, int] = (1280, 1280),
    confidence_threshold: float = 0.2,
    use_matplotlib: bool = False,
    **kwargs
) -> None:
    """
    Save detection visualization to file.
    
    Args:
        image (np.ndarray or PIL.Image): Original image
        detections (List[Dict]): Detection results
        output_path (str): Path to save the visualization
        original_shape (Tuple[int, int], optional): Original image shape (height, width)
        letterbox_shape (Tuple[int, int]): Letterbox shape used during preprocessing
        confidence_threshold (float): Minimum confidence to display
        use_matplotlib (bool): If True, use matplotlib for plotting (higher quality)
        **kwargs: Additional arguments passed to plotting functions
        
    Example:
        >>> save_detection_visualization(
        ...     image, detections, "results.jpg", 
        ...     confidence_threshold=0.5
        ... )
    """
    if use_matplotlib:
        fig = plot_detections_matplotlib(
            image, detections, original_shape, letterbox_shape, 
            confidence_threshold, **kwargs
        )
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        result_image = plot_detections_pil(
            image, detections, original_shape, letterbox_shape, 
            confidence_threshold, **kwargs
        )
        result_image.save(output_path)
