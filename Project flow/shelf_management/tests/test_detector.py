import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.models.detector import ShelfDetector
from src.utils.validation import (
    validate_detection_parameters,
    validate_image,
    validate_image_size,
    validate_model_path,
    validate_planogram
)

@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    return img

@pytest.fixture
def sample_planogram():
    """Create a sample planogram for testing."""
    return {
        'products': ['product1', 'product2', 'product3'],
        'positions': {
            'product1': [0.1, 0.2, 0.3, 0.4],
            'product2': [0.5, 0.6, 0.7, 0.8],
            'product3': [0.9, 1.0, 1.1, 1.2]
        },
        'facings': {
            'product1': 2,
            'product2': 3,
            'product3': 1
        }
    }

@pytest.fixture
def detector():
    """Create a detector instance for testing."""
    model_path = "models/yolov5s.pt"
    return ShelfDetector(
        model_path=model_path,
        conf_threshold=0.25,
        iou_threshold=0.45
    )

def test_validate_image(sample_image):
    """Test image validation."""
    # Test numpy array
    validated_img = validate_image(sample_image)
    assert isinstance(validated_img, np.ndarray)
    assert validated_img.shape == (640, 640, 3)
    
    # Test PIL Image
    pil_img = Image.fromarray(sample_image)
    validated_img = validate_image(pil_img)
    assert isinstance(validated_img, np.ndarray)
    assert validated_img.shape == (640, 640, 3)
    
    # Test invalid input
    with pytest.raises(ValueError):
        validate_image(None)
    
    with pytest.raises(ValueError):
        validate_image(np.random.rand(640, 640))  # 2D array

def test_validate_model_path():
    """Test model path validation."""
    # Test valid path
    validate_model_path("models/model.pt")
    
    # Test invalid paths
    with pytest.raises(ValueError):
        validate_model_path("")
    
    with pytest.raises(ValueError):
        validate_model_path("models/model<.pt")
    
    with pytest.raises(ValueError):
        validate_model_path(123)  # Not a string

def test_validate_planogram(sample_planogram):
    """Test planogram validation."""
    # Test valid planogram
    validate_planogram(sample_planogram)
    
    # Test invalid planograms
    with pytest.raises(ValueError):
        validate_planogram({})  # Missing keys
    
    with pytest.raises(ValueError):
        validate_planogram({
            'products': 'not a list',
            'positions': {},
            'facings': {}
        })
    
    with pytest.raises(ValueError):
        validate_planogram({
            'products': [],
            'positions': 'not a dict',
            'facings': {}
        })

def test_validate_detection_parameters():
    """Test detection parameters validation."""
    # Test valid parameters
    validate_detection_parameters(0.5, 0.5)
    
    # Test invalid parameters
    with pytest.raises(ValueError):
        validate_detection_parameters(-0.1, 0.5)
    
    with pytest.raises(ValueError):
        validate_detection_parameters(0.5, 1.5)

def test_validate_image_size(sample_image):
    """Test image size validation."""
    # Test valid size
    validate_image_size(sample_image)
    
    # Test invalid sizes
    small_img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        validate_image_size(small_img)
    
    large_img = np.random.randint(0, 255, (5000, 5000, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        validate_image_size(large_img)

def test_detector_initialization():
    """Test detector initialization."""
    model_path = "models/yolov5s.pt"
    detector = ShelfDetector(
        model_path=model_path,
        conf_threshold=0.25,
        iou_threshold=0.45
    )
    
    assert detector.model_path == model_path
    assert detector.conf_threshold == 0.25
    assert detector.iou_threshold == 0.45
    assert isinstance(detector.model, torch.nn.Module)

def test_detector_detection(detector, sample_image):
    """Test product detection."""
    results = detector.detect(sample_image)
    
    assert isinstance(results, dict)
    assert 'detections' in results
    assert isinstance(results['detections'], list)
    
    # Test with return_image=True
    results_with_image = detector.detect(
        sample_image,
        return_image=True
    )
    assert 'annotated_image' in results_with_image
    assert isinstance(results_with_image['annotated_image'], np.ndarray)

def test_detector_compliance(detector, sample_image, sample_planogram):
    """Test planogram compliance checking."""
    # First get detections
    results = detector.detect(sample_image)
    
    # Check compliance
    compliance = detector.check_compliance(
        results['detections'],
        sample_planogram
    )
    
    assert isinstance(compliance, dict)
    assert 'compliance_score' in compliance
    assert 'missing_products' in compliance
    assert 'extra_products' in compliance
    assert 'misplaced_products' in compliance

def test_detector_save_load(detector, tmp_path):
    """Test model saving and loading."""
    # Save model
    save_path = tmp_path / "model.pt"
    detector.save_model(str(save_path))
    assert save_path.exists()
    
    # Load model
    new_detector = ShelfDetector.load_model(str(save_path))
    assert isinstance(new_detector, ShelfDetector)
    assert new_detector.model_path == str(save_path) 