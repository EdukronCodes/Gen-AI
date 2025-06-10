import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ..utils.image_utils import preprocess_image, postprocess_detections
from ..utils.validation import validate_image, validate_model_path

logger = logging.getLogger(__name__)

class ShelfDetector:
    """Detects products and analyzes shelf conditions using YOLOv5."""

    def __init__(
        self,
        model_path: str = "yolov5s.pt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        """
        Initialize the shelf detector.

        Args:
            model_path: Path to the YOLOv5 model weights
            device: Device to run the model on ('cuda' or 'cpu')
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        try:
            # Load YOLOv5 model
            self.model = torch.hub.load(
                'ultralytics/yolov5',
                'custom',
                path=model_path,
                force_reload=True
            )
            self.model.to(device)
            self.model.conf = conf_threshold
            self.model.iou = iou_threshold
            logger.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def detect(
        self,
        image: Union[str, np.ndarray, Image.Image],
        return_image: bool = False
    ) -> Dict[str, Union[List[Dict], np.ndarray]]:
        """
        Detect products in a shelf image.

        Args:
            image: Input image (path, numpy array, or PIL Image)
            return_image: Whether to return the annotated image

        Returns:
            Dictionary containing detections and optionally the annotated image
        """
        try:
            # Validate and preprocess image
            image = validate_image(image)
            processed_image = preprocess_image(image)

            # Run inference
            results = self.model(processed_image)

            # Process detections
            detections = postprocess_detections(results)

            # Prepare response
            response = {"detections": detections}

            if return_image:
                # Draw detections on image
                annotated_image = results.render()[0]
                response["annotated_image"] = annotated_image

            return response

        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            raise

    def analyze_shelf(
        self,
        image: Union[str, np.ndarray, Image.Image],
        planogram: Optional[Dict] = None
    ) -> Dict:
        """
        Analyze shelf conditions and compliance.

        Args:
            image: Input image
            planogram: Optional planogram data for compliance checking

        Returns:
            Dictionary containing shelf analysis results
        """
        # Get detections
        detections = self.detect(image)["detections"]

        # Initialize analysis results
        analysis = {
            "total_products": len(detections),
            "out_of_stock": [],
            "misplaced_products": [],
            "compliance_score": 0.0
        }

        if planogram:
            # Check planogram compliance
            analysis.update(self._check_compliance(detections, planogram))

        return analysis

    def _check_compliance(
        self,
        detections: List[Dict],
        planogram: Dict
    ) -> Dict:
        """
        Check shelf compliance against planogram.

        Args:
            detections: List of detected products
            planogram: Planogram data

        Returns:
            Dictionary containing compliance analysis
        """
        compliance = {
            "compliance_score": 0.0,
            "misplaced_products": [],
            "missing_products": [],
            "extra_products": []
        }

        # Compare detections with planogram
        expected_products = set(planogram["products"])
        detected_products = {d["class"] for d in detections}

        # Find missing products
        compliance["missing_products"] = list(
            expected_products - detected_products
        )

        # Find extra products
        compliance["extra_products"] = list(
            detected_products - expected_products
        )

        # Calculate compliance score
        if expected_products:
            compliance["compliance_score"] = len(
                expected_products & detected_products
            ) / len(expected_products)

        return compliance

    def save_model(self, path: str) -> None:
        """
        Save the model weights.

        Args:
            path: Directory path to save the model
        """
        try:
            validate_model_path(path)
            self.model.save(path)
            logger.info(f"Model saved successfully to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    @classmethod
    def load_model(cls, path: str, **kwargs) -> "ShelfDetector":
        """
        Load a saved model.

        Args:
            path: Directory path containing the saved model
            **kwargs: Additional arguments for initialization

        Returns:
            ShelfDetector instance with loaded model
        """
        try:
            validate_model_path(path)
            instance = cls(**kwargs)
            instance.model = torch.hub.load(
                'ultralytics/yolov5',
                'custom',
                path=path,
                force_reload=True
            )
            return instance
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise 