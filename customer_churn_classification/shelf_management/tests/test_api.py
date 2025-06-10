import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.app import app

@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)

@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img)
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

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

def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_analyze_endpoint(client, sample_image, sample_planogram):
    """Test analyze endpoint."""
    # Test without planogram
    response = client.post(
        "/analyze",
        files={"image": ("test.png", sample_image, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    
    # Test with planogram
    response = client.post(
        "/analyze",
        files={"image": ("test.png", sample_image, "image/png")},
        json={"planogram": sample_planogram}
    )
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert "compliance_score" in data
    assert "missing_products" in data
    assert "extra_products" in data
    assert "misplaced_products" in data

def test_detect_endpoint(client, sample_image):
    """Test detect endpoint."""
    # Test with default parameters
    response = client.post(
        "/detect",
        files={"image": ("test.png", sample_image, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    
    # Test with custom parameters
    response = client.post(
        "/detect",
        files={"image": ("test.png", sample_image, "image/png")},
        params={
            "conf_threshold": 0.3,
            "iou_threshold": 0.5,
            "return_image": True
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert "annotated_image" in data

def test_invalid_image(client):
    """Test with invalid image."""
    # Test with empty file
    response = client.post(
        "/detect",
        files={"image": ("test.png", b"", "image/png")}
    )
    assert response.status_code == 500
    
    # Test with invalid file type
    response = client.post(
        "/detect",
        files={"image": ("test.txt", b"invalid", "text/plain")}
    )
    assert response.status_code == 500

def test_invalid_planogram(client, sample_image):
    """Test with invalid planogram."""
    invalid_planogram = {
        'products': 'not a list',
        'positions': {},
        'facings': {}
    }
    
    response = client.post(
        "/analyze",
        files={"image": ("test.png", sample_image, "image/png")},
        json={"planogram": invalid_planogram}
    )
    assert response.status_code == 500

def test_invalid_parameters(client, sample_image):
    """Test with invalid parameters."""
    # Test with invalid confidence threshold
    response = client.post(
        "/detect",
        files={"image": ("test.png", sample_image, "image/png")},
        params={"conf_threshold": 1.5}
    )
    assert response.status_code == 500
    
    # Test with invalid IoU threshold
    response = client.post(
        "/detect",
        files={"image": ("test.png", sample_image, "image/png")},
        params={"iou_threshold": -0.1}
    )
    assert response.status_code == 500 