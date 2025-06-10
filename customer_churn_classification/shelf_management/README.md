# Computer Vision for Retail Shelf Management

This project implements a computer vision system for monitoring and analyzing retail shelf conditions using deep learning.

## Project Overview

### Problem Definition
- **Business Problem**: Monitor and analyze retail shelf conditions in real-time
- **Type**: Computer Vision / Object Detection
- **Success Metrics**: 
  - Detection accuracy
  - Processing speed
  - Shelf compliance rate
  - Out-of-stock detection rate

### Objectives
- Detect product presence and positioning
- Identify out-of-stock situations
- Monitor planogram compliance
- Track product facings
- Detect price tag issues

### Constraints
- Real-time processing (< 1 second per image)
- Support for multiple camera angles
- Low-light condition handling
- Scalable to multiple store locations

## Technical Implementation

### Data Sources
- Shelf images from store cameras
- Product catalog database
- Planogram data
- Historical shelf images

### Key Technologies
- YOLOv5 for object detection
- OpenCV for image processing
- FastAPI for API deployment
- Redis for caching
- Docker for containerization

## Project Structure
```
shelf_management/
├── data/
│   ├── raw/              # Raw images
│   ├── processed/        # Processed images
│   └── models/          # Trained models
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   └── preprocessor.py
│   ├── models/
│   │   ├── detector.py
│   │   └── classifier.py
│   ├── utils/
│   │   ├── image_utils.py
│   │   └── validation.py
│   └── api/
│       └── app.py
├── tests/
│   └── test_detector.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Setup and Installation

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Preparation:
```bash
python src/data/data_loader.py
```

2. Model Training:
```bash
python src/models/train.py
```

3. Run API:
```bash
python src/api/app.py
```

## API Documentation

The API provides endpoints for:
- Image processing
- Shelf analysis
- Compliance checking
- Out-of-stock detection

## Monitoring

- Detection accuracy metrics
- Processing time
- System resource usage
- Error rates

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 