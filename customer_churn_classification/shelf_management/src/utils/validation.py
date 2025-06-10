import logging
from pathlib import Path
from typing import Dict, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

def validate_image(
    image: Union[str, np.ndarray, Image.Image]
) -> np.ndarray:
    """
    Validate input image.

    Args:
        image: Input image (path, numpy array, or PIL Image)

    Returns:
        Validated image as numpy array

    Raises:
        ValueError: If image validation fails
    """
    try:
        # Handle string path
        if isinstance(image, str):
            if not Path(image).exists():
                raise ValueError(f"Image file not found: {image}")
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Failed to load image: {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Handle PIL Image
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        # Validate numpy array
        elif isinstance(image, np.ndarray):
            if image.ndim != 3:
                raise ValueError("Image must be 3-dimensional")
            if image.shape[2] != 3:
                raise ValueError("Image must have 3 color channels")
        else:
            raise ValueError(
                f"Invalid image type: {type(image)}. "
                "Must be string path, numpy array, or PIL Image"
            )
        
        return image

    except Exception as e:
        logger.error(f"Error validating image: {str(e)}")
        raise

def validate_model_path(path: str) -> None:
    """
    Validate model path.

    Args:
        path: Path to validate

    Raises:
        ValueError: If path validation fails
    """
    if not path:
        raise ValueError("Model path cannot be empty")

    if not isinstance(path, str):
        raise ValueError("Model path must be a string")

    # Check for invalid characters
    invalid_chars = {'<', '>', ':', '"', '/', '\\', '|', '?', '*'}
    if any(char in path for char in invalid_chars):
        raise ValueError(f"Model path contains invalid characters: {invalid_chars}")

def validate_planogram(planogram: Dict) -> None:
    """
    Validate planogram data.

    Args:
        planogram: Planogram data to validate

    Raises:
        ValueError: If planogram validation fails
    """
    if not isinstance(planogram, dict):
        raise ValueError("Planogram must be a dictionary")

    required_keys = {'products', 'positions', 'facings'}
    missing_keys = required_keys - set(planogram.keys())
    if missing_keys:
        raise ValueError(f"Missing required keys in planogram: {missing_keys}")

    if not isinstance(planogram['products'], list):
        raise ValueError("'products' must be a list")

    if not isinstance(planogram['positions'], dict):
        raise ValueError("'positions' must be a dictionary")

    if not isinstance(planogram['facings'], dict):
        raise ValueError("'facings' must be a dictionary")

def validate_detection_parameters(
    conf_threshold: float,
    iou_threshold: float
) -> None:
    """
    Validate detection parameters.

    Args:
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold

    Raises:
        ValueError: If parameter validation fails
    """
    if not 0 <= conf_threshold <= 1:
        raise ValueError("conf_threshold must be between 0 and 1")

    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1")

def validate_image_size(
    image: np.ndarray,
    min_size: tuple = (100, 100),
    max_size: tuple = (4000, 4000)
) -> None:
    """
    Validate image dimensions.

    Args:
        image: Input image
        min_size: Minimum allowed dimensions
        max_size: Maximum allowed dimensions

    Raises:
        ValueError: If size validation fails
    """
    height, width = image.shape[:2]
    
    if height < min_size[0] or width < min_size[1]:
        raise ValueError(
            f"Image dimensions ({height}x{width}) below minimum "
            f"allowed size {min_size}"
        )
    
    if height > max_size[0] or width > max_size[1]:
        raise ValueError(
            f"Image dimensions ({height}x{width}) above maximum "
            f"allowed size {max_size}"
        ) 