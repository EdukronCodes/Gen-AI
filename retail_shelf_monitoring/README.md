# Retail Shelf Monitoring using YOLOv8

This project implements a real-time retail shelf monitoring system using YOLOv8 for object detection. It helps track product placement, detect out-of-stock items, and monitor shelf organization.

## Features

- Real-time object detection using YOLOv8
- Product placement monitoring
- Out-of-stock detection
- Shelf organization analysis
- Alert system for shelf issues
- Web dashboard for monitoring
- Historical data tracking
- Multiple camera support

## Project Structure

```
retail_shelf_monitoring/
├── backend/                 # FastAPI backend
│   ├── api/                # API endpoints
│   ├── models/             # Database models
│   ├── services/           # YOLOv8 and monitoring services
│   └── utils/              # Utility functions
├── frontend/               # React frontend
│   ├── public/            # Static files
│   └── src/               # Source code
│       ├── components/    # React components
│       ├── pages/         # Page components
│       ├── services/      # API services
│       └── utils/         # Utility functions
└── README.md              # Project documentation
```

## Technology Stack

### Backend
- Python 3.8+
- FastAPI
- YOLOv8
- OpenCV
- SQLAlchemy
- PostgreSQL
- Redis (for real-time updates)

### Frontend
- React 18
- Material-UI
- Chart.js
- WebSocket
- Axios

## Setup Instructions

1. Clone the repository
2. Set up the backend:
   ```bash
   cd retail_shelf_monitoring/backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```

3. Set up the frontend:
   ```bash
   cd retail_shelf_monitoring/frontend
   npm install
   npm start
   ```

4. Access the application at http://localhost:3000

## API Endpoints

- `POST /api/detect` - Process image/video for object detection
- `GET /api/shelves` - Get shelf status
- `GET /api/alerts` - Get monitoring alerts
- `GET /api/statistics` - Get monitoring statistics
- `WebSocket /ws/monitoring` - Real-time monitoring updates

## Environment Variables

- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `YOLO_MODEL_PATH` - Path to YOLOv8 model
- `CAMERA_URLS` - Comma-separated list of camera URLs

## License

This project is licensed under the MIT License - see the LICENSE file for details. 