import cv2
import numpy as np
from ultralytics import YOLO
import logging
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ShelfMonitor:
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        save_detections: bool = True
    ):
        """
        Initialize the shelf monitoring system.
        
        Args:
            model_path (str): Path to the trained YOLOv8 model
            confidence_threshold (float): Minimum confidence for detections
            iou_threshold (float): IoU threshold for NMS
            save_detections (bool): Whether to save detection images
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.save_detections = save_detections
        
        # Create output directory for saved detections
        if save_detections:
            self.output_dir = Path('detections')
            self.output_dir.mkdir(exist_ok=True)
        
        logger.info("ShelfMonitor initialized successfully")

    def process_frame(
        self,
        frame: np.ndarray,
        camera_id: Optional[int] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single frame for object detection.
        
        Args:
            frame (np.ndarray): Input frame
            camera_id (Optional[int]): ID of the camera
            
        Returns:
            Tuple[np.ndarray, List[Dict]]: Annotated frame and detection results
        """
        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )[0]

            # Process detections
            detections = []
            annotated_frame = frame.copy()

            for box in results.boxes:
                # Get detection details
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = results.names[class_id]

                # Create detection dictionary
                detection = {
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'timestamp': datetime.utcnow().isoformat()
                }
                detections.append(detection)

                # Draw bounding box
                cv2.rectangle(
                    annotated_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )

                # Add label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(
                    annotated_frame,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

            # Save detection if enabled
            if self.save_detections and camera_id is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = self.output_dir / f"camera_{camera_id}_{timestamp}.jpg"
                cv2.imwrite(str(image_path), annotated_frame)

                # Save detection metadata
                metadata_path = self.output_dir / f"camera_{camera_id}_{timestamp}.json"
                with open(metadata_path, 'w') as f:
                    json.dump({
                        'camera_id': camera_id,
                        'timestamp': timestamp,
                        'detections': detections
                    }, f, indent=2)

            return annotated_frame, detections

        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame, []

    def analyze_shelf_status(
        self,
        detections: List[Dict]
    ) -> Dict:
        """
        Analyze shelf status based on detections.
        
        Args:
            detections (List[Dict]): List of detection results
            
        Returns:
            Dict: Shelf status analysis
        """
        try:
            # Count detections by class
            class_counts = {}
            for detection in detections:
                class_name = detection['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            # Calculate metrics
            total_products = class_counts.get('product', 0)
            empty_shelves = class_counts.get('empty_shelf', 0)
            misplaced_products = class_counts.get('misplaced_product', 0)

            # Calculate fill rate
            total_shelf_space = total_products + empty_shelves
            fill_rate = (total_products / total_shelf_space) if total_shelf_space > 0 else 0

            # Calculate organization score
            organization_score = 1.0 - (misplaced_products / total_products) if total_products > 0 else 1.0

            return {
                'total_products': total_products,
                'empty_shelves': empty_shelves,
                'misplaced_products': misplaced_products,
                'fill_rate': fill_rate,
                'organization_score': organization_score,
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error analyzing shelf status: {str(e)}")
            return {}

    def process_video_stream(
        self,
        camera_url: str,
        camera_id: Optional[int] = None,
        display: bool = True
    ):
        """
        Process video stream from a camera.
        
        Args:
            camera_url (str): URL or path to video stream
            camera_id (Optional[int]): ID of the camera
            display (bool): Whether to display the processed frame
        """
        try:
            cap = cv2.VideoCapture(camera_url)
            if not cap.isOpened():
                raise ValueError(f"Could not open video stream: {camera_url}")

            logger.info(f"Processing video stream from {camera_url}")

            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from video stream")
                    break

                # Process frame
                annotated_frame, detections = self.process_frame(frame, camera_id)

                # Analyze shelf status
                shelf_status = self.analyze_shelf_status(detections)
                logger.info(f"Shelf status: {shelf_status}")

                # Display frame if enabled
                if display:
                    cv2.imshow('Shelf Monitoring', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            logger.error(f"Error processing video stream: {str(e)}")
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    model_path = "runs/train/best.pt"
    camera_url = 0  # Use 0 for webcam, or provide RTSP URL for IP camera
    
    monitor = ShelfMonitor(
        model_path=model_path,
        confidence_threshold=0.5,
        iou_threshold=0.45,
        save_detections=True
    )
    
    monitor.process_video_stream(
        camera_url=camera_url,
        camera_id=1,
        display=True
    ) 