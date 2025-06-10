import logging
from typing import Dict, List, Union

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

def preprocess_image(
    image: Union[str, np.ndarray, Image.Image],
    target_size: tuple = (640, 640)
) -> np.ndarray:
    """
    Preprocess image for model input.

    Args:
        image: Input image (path, numpy array, or PIL Image)
        target_size: Target size for resizing

    Returns:
        Preprocessed image as numpy array
    """
    try:
        # Convert to numpy array if needed
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        # Resize image
        image = cv2.resize(image, target_size)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        return image

    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def postprocess_detections(results) -> List[Dict]:
    """
    Process model detection results.

    Args:
        results: Raw detection results from model

    Returns:
        List of processed detections
    """
    try:
        detections = []
        
        # Convert results to numpy array
        pred = results.xyxy[0].cpu().numpy()
        
        for *xyxy, conf, cls in pred:
            detection = {
                "bbox": {
                    "x1": float(xyxy[0]),
                    "y1": float(xyxy[1]),
                    "x2": float(xyxy[2]),
                    "y2": float(xyxy[3])
                },
                "confidence": float(conf),
                "class": int(cls),
                "class_name": results.names[int(cls)]
            }
            detections.append(detection)
        
        return detections

    except Exception as e:
        logger.error(f"Error postprocessing detections: {str(e)}")
        raise

def draw_detections(
    image: np.ndarray,
    detections: List[Dict],
    class_names: Dict[int, str]
) -> np.ndarray:
    """
    Draw detection boxes and labels on image.

    Args:
        image: Input image
        detections: List of detections
        class_names: Dictionary mapping class IDs to names

    Returns:
        Image with detections drawn
    """
    try:
        # Create a copy of the image
        output = image.copy()
        
        # Define colors for different classes
        colors = {
            cls_id: (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            )
            for cls_id in class_names.keys()
        }
        
        for detection in detections:
            # Get box coordinates
            x1, y1 = int(detection["bbox"]["x1"]), int(detection["bbox"]["y1"])
            x2, y2 = int(detection["bbox"]["x2"]), int(detection["bbox"]["y2"])
            
            # Get class info
            cls_id = detection["class"]
            cls_name = class_names[cls_id]
            conf = detection["confidence"]
            
            # Draw box
            color = colors[cls_id]
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{cls_name}: {conf:.2f}"
            cv2.putText(
                output,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        return output

    except Exception as e:
        logger.error(f"Error drawing detections: {str(e)}")
        raise

def calculate_iou(box1: Dict, box2: Dict) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.

    Args:
        box1: First bounding box
        box2: Second bounding box

    Returns:
        IoU score
    """
    try:
        # Get coordinates
        x1 = max(box1["bbox"]["x1"], box2["bbox"]["x1"])
        y1 = max(box1["bbox"]["y1"], box2["bbox"]["y1"])
        x2 = min(box1["bbox"]["x2"], box2["bbox"]["x2"])
        y2 = min(box1["bbox"]["y2"], box2["bbox"]["y2"])
        
        # Calculate intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate union area
        box1_area = (
            (box1["bbox"]["x2"] - box1["bbox"]["x1"]) *
            (box1["bbox"]["y2"] - box1["bbox"]["y1"])
        )
        box2_area = (
            (box2["bbox"]["x2"] - box2["bbox"]["x1"]) *
            (box2["bbox"]["y2"] - box2["bbox"]["y1"])
        )
        union = box1_area + box2_area - intersection
        
        # Calculate IoU
        iou = intersection / union if union > 0 else 0
        
        return iou

    except Exception as e:
        logger.error(f"Error calculating IoU: {str(e)}")
        raise

def apply_augmentation(
    image: np.ndarray,
    augmentation_type: str = "none"
) -> np.ndarray:
    """
    Apply image augmentation.

    Args:
        image: Input image
        augmentation_type: Type of augmentation to apply

    Returns:
        Augmented image
    """
    try:
        if augmentation_type == "none":
            return image
            
        # Define augmentation pipeline
        if augmentation_type == "basic":
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2
                ),
                transforms.ToTensor()
            ])
        else:
            raise ValueError(f"Unknown augmentation type: {augmentation_type}")
        
        # Apply augmentation
        augmented = transform(image)
        
        return augmented.numpy()

    except Exception as e:
        logger.error(f"Error applying augmentation: {str(e)}")
        raise 