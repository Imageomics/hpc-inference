"""
Image transformation utilities for computer vision models.

This module contains preprocessing transforms adapted from various sources,
primarily for object detection models like YOLO and MegaDetector.

Copyright attributions are included for each function/class as appropriate.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image
from typing import Union, Tuple


def letterbox(
    im: Union[Image.Image, torch.Tensor], 
    new_shape: Union[int, Tuple[int, int]] = (640, 640), 
    color: Tuple[int, int, int] = (114, 114, 114), 
    auto: bool = False, 
    scaleFill: bool = False, 
    scaleup: bool = True, 
    stride: int = 32
) -> torch.Tensor:
    """
    Resize and pad an image to a desired shape while keeping the aspect ratio unchanged.

    This function is commonly used in object detection tasks to prepare images for models 
    like YOLOv5 and MegaDetector. It resizes the image to fit into the new shape with the 
    correct aspect ratio and then pads the rest with a specified color.

    Based on letterbox implementation from Microsoft CameraTraps PytorchWildlife.
    Copyright (c) Microsoft Corporation. All rights reserved.
    Licensed under the MIT License.
    
    Original source: https://github.com/microsoft/CameraTraps/blob/main/PytorchWildlife/data/transforms.py

    Args:
        im (PIL.Image.Image or torch.Tensor): The input image. Can be a PIL image or 
            a PyTorch tensor with shape (C, H, W) and values in [0, 1].
        new_shape (int or tuple, optional): The target size of the image. If int, creates
            a square image (new_shape, new_shape). If tuple, should be (height, width).
            Defaults to (640, 640).
        color (tuple, optional): The RGB color values used for padding, range [0, 255].
            Defaults to (114, 114, 114) which is a gray color commonly used in YOLO.
        auto (bool, optional): If True, adjusts padding to ensure the padded image 
            dimensions are a multiple of the stride. Defaults to False.
        scaleFill (bool, optional): If True, scales the image to fill the new shape 
            exactly, ignoring aspect ratio (may cause distortion). Defaults to False.
        scaleup (bool, optional): If True, allows the function to scale up the image.
            If False, only scales down. Defaults to True.
        stride (int, optional): The stride used in the model. When auto=True, padding 
            is adjusted to be a multiple of this stride. Defaults to 32.

    Returns:
        torch.Tensor: The transformed image as a tensor with shape (C, H, W) and 
            values in [0, 1]. The output will have exactly the dimensions specified 
            by new_shape.

    Examples:
        >>> from PIL import Image
        >>> import torch
        >>> 
        >>> # Load and transform a PIL image
        >>> pil_img = Image.open("photo.jpg")  # e.g., 800x600 RGB image
        >>> transformed = letterbox(pil_img, new_shape=640)
        >>> print(transformed.shape)  # torch.Size([3, 640, 640])
        >>> 
        >>> # Transform a tensor image
        >>> tensor_img = torch.rand(3, 480, 320)  # Random image tensor
        >>> transformed = letterbox(tensor_img, new_shape=(1280, 1280), scaleup=False)
        >>> print(transformed.shape)  # torch.Size([3, 1280, 1280])
        >>> 
        >>> # Use with auto padding for model stride
        >>> transformed = letterbox(pil_img, new_shape=640, auto=True, stride=32)
        >>> # Output dimensions will be multiples of 32

    Note:
        - Input PIL images are automatically converted to tensors
        - The function preserves aspect ratio unless scaleFill=True
        - Padding is applied symmetrically when possible
        - Output tensor values are normalized to [0, 1] range
    """
    # Convert PIL Image to Torch Tensor
    if isinstance(im, Image.Image):
        im = T.ToTensor()(im)

    # Original shape
    shape = im.shape[1:]  # shape = [height, width]

    # New shape - convert int to tuple if needed
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old) and compute padding
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        # Make padding a multiple of stride
        dw, dh = dw % stride, dh % stride
    elif scaleFill:
        # Scale to fill entire new_shape, ignore aspect ratio
        dw, dh = 0, 0
        new_unpad = new_shape
        r = new_shape[1] / shape[1], new_shape[0] / shape[0]

    # Divide padding by 2 for symmetric padding
    dw /= 2
    dh /= 2
   
    # Resize image if current size != target unpadded size
    if shape[::-1] != new_unpad:
        resize_transform = T.Resize(
            new_unpad[::-1], 
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=False
        )
        im = resize_transform(im)

    # Apply padding
    # Padding format: (left, right, top, bottom)
    padding = (
        int(round(dw - 0.1)), 
        int(round(dw + 0.1)), 
        int(round(dh - 0.1)), 
        int(round(dh + 0.1))
    )
    # Scale to [0,255], pad with color value, then scale back to [0,1]
    im = F.pad(im * 255.0, padding, value=color[0]) / 255.0

    return im


class MegaDetector_v5_Transform:
    """
    A transformation class to preprocess images for the MegaDetector v5 model.
    
    This transform handles the complete preprocessing pipeline required for MegaDetector,
    including image format conversion, tensor conversion, normalization, and the specific
    letterbox resizing used by YOLO-based detection models.

    Based on Microsoft CameraTraps PytorchWildlife transforms.
    Copyright (c) Microsoft Corporation. All rights reserved.
    Licensed under the MIT License.
    
    Original source: https://github.com/microsoft/CameraTraps/blob/main/PytorchWildlife/data/transforms.py

    This transformation is specifically designed for YOLOv5-based MegaDetector models
    and ensures proper preprocessing for optimal detection performance.

    Attributes:
        target_size (int): The target size for the image's longest side after resizing.
        stride (int): Stride value used for padding calculations in letterbox transform.

    Examples:
        >>> from PIL import Image
        >>> import numpy as np
        >>> 
        >>> # Create transform for MegaDetector
        >>> transform = MegaDetector_v5_Transform(target_size=1280, stride=32)
        >>> 
        >>> # Transform a PIL image
        >>> pil_img = Image.open("wildlife_photo.jpg")
        >>> tensor_output = transform(pil_img)
        >>> print(tensor_output.shape)  # torch.Size([3, 1280, 1280])
        >>> 
        >>> # Transform a numpy array
        >>> np_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        >>> tensor_output = transform(np_img)
        >>> print(tensor_output.shape)  # torch.Size([3, 1280, 1280])
        >>> 
        >>> # Use in a data pipeline
        >>> from torchvision.datasets import ImageFolder
        >>> dataset = ImageFolder(root='images/', transform=transform)

    Note:
        - Handles both PIL Images and numpy arrays as input
        - Always outputs a torch.Tensor with values in [0, 1] range
        - Preserves aspect ratio using letterbox padding
        - Optimized for MegaDetector model input requirements
    """

    def __init__(self, target_size: int = 1280, stride: int = 32):
        """
        Initialize the MegaDetector v5 transform.

        Args:
            target_size (int, optional): Desired size for the image's square output 
                dimensions. MegaDetector typically uses 1280x1280. Defaults to 1280.
            stride (int, optional): Stride value for letterbox padding calculations.
                Should match the model's architectural stride. Defaults to 32.

        Examples:
            >>> # Standard MegaDetector transform
            >>> transform = MegaDetector_v5_Transform()
            >>> 
            >>> # Custom size for different model variants
            >>> transform = MegaDetector_v5_Transform(target_size=640, stride=32)
            >>> 
            >>> # High resolution for better accuracy
            >>> transform = MegaDetector_v5_Transform(target_size=1920, stride=64)
        """
        self.target_size = target_size
        self.stride = stride

    def __call__(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Apply the transformation to the provided image.

        This method handles the complete preprocessing pipeline:
        1. Convert PIL Image to numpy array if needed
        2. Convert numpy array to torch tensor with proper channel ordering
        3. Normalize pixel values to [0, 1] range
        4. Apply letterbox transform for aspect-ratio preserving resize and padding

        Args:
            image (np.ndarray or PIL.Image): Input image. Can be:
                - PIL Image in any mode (will be converted to RGB)
                - numpy array with shape (H, W, C) and dtype uint8 or float
                - Values should be in range [0, 255] for uint8 or [0, 1] for float

        Returns:
            torch.Tensor: Transformed image tensor with shape (C, H, W) where:
                - C = 3 (RGB channels)
                - H = W = target_size
                - Values are in range [0, 1]
                - Aspect ratio is preserved with gray padding

        Raises:
            TypeError: If input image is not a PIL Image or numpy array
            ValueError: If numpy array doesn't have the expected shape

        Examples:
            >>> transform = MegaDetector_v5_Transform(target_size=640)
            >>> 
            >>> # PIL Image input
            >>> pil_img = Image.open("animal.jpg")  # e.g., 1200x800 image
            >>> result = transform(pil_img)
            >>> print(result.shape, result.dtype, result.min(), result.max())
            >>> # torch.Size([3, 640, 640]) torch.float32 tensor(0.) tensor(1.)
            >>> 
            >>> # Numpy array input
            >>> np_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> result = transform(np_img)
            >>> print(result.shape)  # torch.Size([3, 640, 640])
            >>> 
            >>> # Grayscale PIL image (automatically converted to RGB)
            >>> gray_img = Image.open("grayscale.jpg").convert('L')
            >>> result = transform(gray_img.convert('RGB'))
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            # Ensure RGB mode for consistent 3-channel output
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = np.array(image)
        elif not isinstance(image, np.ndarray):
            raise TypeError(f"Expected PIL Image or numpy array, got {type(image)}")
            
        # Convert numpy array to PyTorch tensor with proper formatting
        if isinstance(image, np.ndarray):
            # Validate shape
            if image.ndim != 3:
                raise ValueError(f"Expected 3D array (H, W, C), got shape {image.shape}")
            if image.shape[2] != 3:
                raise ValueError(f"Expected 3 channels (RGB), got {image.shape[2]}")
                
            # Convert from HWC to CHW format
            image = image.transpose((2, 0, 1))
            image = np.ascontiguousarray(image)
            image = torch.from_numpy(image).float()
            
            # Normalize to [0, 1] range if needed
            if image.max() > 1.0:
                image /= 255.0

        # Apply letterbox transform for aspect-ratio preserving resize and padding
        transformed_image = letterbox(
            image, 
            new_shape=self.target_size, 
            stride=self.stride, 
            auto=False
        )

        return transformed_image

    def __repr__(self) -> str:
        """String representation of the transform."""
        return (f"{self.__class__.__name__}("
                f"target_size={self.target_size}, "
                f"stride={self.stride})")
