from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import cv2
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import logging
from typing import List, Dict, Optional
import asyncio
from models.inference import ShelfMonitor
from models.database import Camera, Product, Detection, Alert, ShelfStatus
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Retail Shelf Monitoring API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./shelf_monitoring.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Initialize shelf monitor
MODEL_PATH = os.getenv("MODEL_PATH", "models/runs/train/best.pt")
shelf_monitor = ShelfMonitor(
    model_path=MODEL_PATH,
    confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.5")),
    iou_threshold=float(os.getenv("IOU_THRESHOLD", "0.45")),
    save_detections=True
)

# Camera streams
active_streams: Dict[int, asyncio.Task] = {}

def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/cameras/")
async def create_camera(
    name: str,
    url: str,
    location: str,
    db: Session = Depends(get_db)
):
    """Create a new camera."""
    try:
        camera = Camera(name=name, url=url, location=location)
        db.add(camera)
        db.commit()
        db.refresh(camera)
        return camera
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/cameras/")
async def get_cameras(db: Session = Depends(get_db)):
    """Get all cameras."""
    return db.query(Camera).all()

@app.post("/cameras/{camera_id}/start")
async def start_camera_stream(
    camera_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start processing stream from a camera."""
    try:
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")

        if camera_id in active_streams:
            raise HTTPException(status_code=400, detail="Stream already active")

        async def process_stream():
            try:
                cap = cv2.VideoCapture(camera.url)
                if not cap.isOpened():
                    raise ValueError(f"Could not open video stream: {camera.url}")

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning("Failed to read frame from video stream")
                        break

                    # Process frame
                    annotated_frame, detections = shelf_monitor.process_frame(frame, camera_id)

                    # Save detections to database
                    for detection in detections:
                        db_detection = Detection(
                            camera_id=camera_id,
                            confidence=detection['confidence'],
                            bounding_box=detection['bbox'],
                            timestamp=datetime.fromisoformat(detection['timestamp'])
                        )
                        db.add(db_detection)

                    # Analyze shelf status
                    shelf_status = shelf_monitor.analyze_shelf_status(detections)
                    db_status = ShelfStatus(
                        camera_id=camera_id,
                        status=shelf_status,
                        timestamp=datetime.utcnow()
                    )
                    db.add(db_status)

                    # Check for alerts
                    if shelf_status.get('fill_rate', 1.0) < 0.7:
                        alert = Alert(
                            camera_id=camera_id,
                            alert_type="low_stock",
                            message="Low stock level detected",
                            severity="high"
                        )
                        db.add(alert)

                    db.commit()
                    await asyncio.sleep(0.1)  # Prevent overwhelming the system

            except Exception as e:
                logger.error(f"Error in stream processing: {str(e)}")
            finally:
                if 'cap' in locals():
                    cap.release()
                active_streams.pop(camera_id, None)

        # Start stream processing in background
        task = asyncio.create_task(process_stream())
        active_streams[camera_id] = task

        return {"message": "Stream processing started"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cameras/{camera_id}/stop")
async def stop_camera_stream(camera_id: int):
    """Stop processing stream from a camera."""
    if camera_id not in active_streams:
        raise HTTPException(status_code=400, detail="Stream not active")

    task = active_streams[camera_id]
    task.cancel()
    active_streams.pop(camera_id)

    return {"message": "Stream processing stopped"}

@app.get("/cameras/{camera_id}/status")
async def get_camera_status(
    camera_id: int,
    db: Session = Depends(get_db)
):
    """Get current status of a camera."""
    try:
        # Get latest shelf status
        status = db.query(ShelfStatus)\
            .filter(ShelfStatus.camera_id == camera_id)\
            .order_by(ShelfStatus.timestamp.desc())\
            .first()

        # Get active alerts
        alerts = db.query(Alert)\
            .filter(
                Alert.camera_id == camera_id,
                Alert.is_resolved == False
            )\
            .all()

        return {
            "shelf_status": status.status if status else None,
            "active_alerts": alerts,
            "stream_active": camera_id in active_streams
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/products/")
async def create_product(
    name: str,
    sku: str,
    category: str,
    expected_position: Dict,
    min_stock_level: int,
    db: Session = Depends(get_db)
):
    """Create a new product."""
    try:
        product = Product(
            name=name,
            sku=sku,
            category=category,
            expected_position=expected_position,
            min_stock_level=min_stock_level
        )
        db.add(product)
        db.commit()
        db.refresh(product)
        return product
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/products/")
async def get_products(db: Session = Depends(get_db)):
    """Get all products."""
    return db.query(Product).all()

@app.post("/detections/upload")
async def upload_detection(
    file: UploadFile = File(...),
    camera_id: int = None,
    db: Session = Depends(get_db)
):
    """Upload and process a single image."""
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process frame
        annotated_frame, detections = shelf_monitor.process_frame(frame, camera_id)

        # Save detections to database
        for detection in detections:
            db_detection = Detection(
                camera_id=camera_id,
                confidence=detection['confidence'],
                bounding_box=detection['bbox'],
                timestamp=datetime.fromisoformat(detection['timestamp'])
            )
            db.add(db_detection)

        # Analyze shelf status
        shelf_status = shelf_monitor.analyze_shelf_status(detections)
        db_status = ShelfStatus(
            camera_id=camera_id,
            status=shelf_status,
            timestamp=datetime.utcnow()
        )
        db.add(db_status)

        db.commit()

        return {
            "detections": detections,
            "shelf_status": shelf_status
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts/")
async def get_alerts(
    resolved: bool = False,
    db: Session = Depends(get_db)
):
    """Get all alerts."""
    return db.query(Alert)\
        .filter(Alert.is_resolved == resolved)\
        .order_by(Alert.created_at.desc())\
        .all()

@app.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: int,
    db: Session = Depends(get_db)
):
    """Mark an alert as resolved."""
    try:
        alert = db.query(Alert).filter(Alert.id == alert_id).first()
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")

        alert.is_resolved = True
        alert.resolved_at = datetime.utcnow()
        db.commit()

        return {"message": "Alert resolved successfully"}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 