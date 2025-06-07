# Sentiment Analysis of Product Reviews

This project implements a sentiment analysis system for product reviews using natural language processing techniques. It provides a web interface for users to analyze reviews and visualize sentiment trends.

## Features

- Review sentiment classification (Positive, Negative, Neutral)
- Sentiment score calculation
- Review submission and analysis
- Sentiment trends visualization
- Product category analysis
- Review keyword extraction
- Sentiment distribution charts
- Review search and filtering

## Project Structure

```
sentiment_analysis/
├── backend/                 # Django backend
│   ├── api/                # API endpoints
│   ├── models/             # Database models
│   ├── services/           # Business logic
│   ├── utils/              # Utility functions
│   └── sentiment/          # Django project settings
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
- Django 4.2
- Django REST Framework
- NLTK
- scikit-learn
- pandas
- numpy
- SQLite (development) / PostgreSQL (production)

### Frontend
- React 18
- Material-UI
- Chart.js
- Axios
- React Router

## Setup Instructions

1. Clone the repository
2. Set up the backend:
   ```bash
   cd sentiment_analysis/backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   python manage.py migrate
   python manage.py runserver
   ```

3. Set up the frontend:
   ```bash
   cd sentiment_analysis/frontend
   npm install
   npm start
   ```

4. Access the application at http://localhost:3000

## API Endpoints

- `POST /api/reviews/` - Submit a new review
- `GET /api/reviews/` - List all reviews
- `GET /api/reviews/{id}/` - Get review details
- `GET /api/reviews/sentiment-stats/` - Get sentiment statistics
- `GET /api/reviews/category-stats/` - Get category-wise statistics
- `GET /api/reviews/keywords/` - Get common keywords

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 