# Customer Churn Classification for Online Retailer

This project implements a machine learning system to predict customer churn for an online retailer. It uses customer behavior data to identify patterns that indicate a customer is likely to stop using the service.

## Features

- Customer behavior analysis
- Churn prediction using machine learning
- Real-time churn risk scoring
- Customer retention recommendations
- Interactive dashboard for monitoring
- API endpoints for integration

## Tech Stack

- Backend: Python, Django, Django REST Framework
- Frontend: React, Material-UI
- Database: PostgreSQL
- Machine Learning: Scikit-learn, Pandas, NumPy
- Visualization: Plotly, Chart.js

## Project Structure

```
customer_churn_classification/
├── backend/
│   ├── api/
│   │   ├── views.py
│   │   ├── urls.py
│   │   └── serializers.py
│   ├── models/
│   │   ├── customer.py
│   │   └── transaction.py
│   ├── services/
│   │   ├── churn_predictor.py
│   │   └── data_processor.py
│   ├── ml_models/
│   │   ├── train_model.py
│   │   └── predict.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── utils/
│   └── package.json
└── data/
    ├── raw/
    └── processed/
```

## Setup Instructions

1. Clone the repository
2. Set up the backend:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   python manage.py migrate
   python manage.py runserver
   ```

3. Set up the frontend:
   ```bash
   cd frontend
   npm install
   npm start
   ```

4. Train the model:
   ```bash
   cd backend
   python ml_models/train_model.py
   ```

## API Endpoints

- `GET /api/customers/` - List all customers
- `GET /api/customers/{id}/` - Get customer details
- `POST /api/customers/{id}/predict-churn/` - Predict churn probability
- `GET /api/customers/churn-stats/` - Get churn statistics

## Model Features

The churn prediction model uses the following features:
- Purchase frequency
- Average order value
- Days since last purchase
- Customer lifetime value
- Product category preferences
- Customer service interactions
- Website engagement metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License. 