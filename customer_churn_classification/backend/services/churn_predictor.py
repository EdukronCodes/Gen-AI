import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from django.conf import settings
from ..models.customer import Customer

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = os.path.join(settings.BASE_DIR, 'ml_models', 'churn_model.joblib')
        self.scaler_path = os.path.join(settings.BASE_DIR, 'ml_models', 'scaler.joblib')
        self.load_model()
    
    def load_model(self):
        """Load the trained model and scaler if they exist"""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
        except:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
    
    def prepare_features(self, customer):
        """Prepare customer features for prediction"""
        features = {
            'days_since_last_purchase': customer.days_since_last_purchase,
            'purchase_frequency': customer.purchase_frequency,
            'average_order_value': float(customer.average_order_value),
            'total_purchases': customer.total_purchases,
            'total_spent': float(customer.total_spent),
            'customer_lifetime_value': float(customer.customer_lifetime_value),
            'website_visits': customer.website_visits,
            'support_tickets': customer.support_tickets,
            'product_returns': customer.product_returns,
        }
        return pd.DataFrame([features])
    
    def predict_churn(self, customer):
        """Predict churn probability for a customer"""
        if not self.model:
            return 0.0
        
        features = self.prepare_features(customer)
        features_scaled = self.scaler.transform(features)
        churn_probability = self.model.predict_proba(features_scaled)[0][1]
        return float(churn_probability)
    
    def train_model(self, customers):
        """Train the churn prediction model"""
        # Prepare training data
        X = []
        y = []
        
        for customer in customers:
            features = self.prepare_features(customer)
            X.append(features.iloc[0].values)
            y.append(1 if customer.is_churned else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Save model and scaler
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        # Return model performance metrics
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score
        }
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        if not self.model:
            return {}
        
        feature_names = [
            'days_since_last_purchase',
            'purchase_frequency',
            'average_order_value',
            'total_purchases',
            'total_spent',
            'customer_lifetime_value',
            'website_visits',
            'support_tickets',
            'product_returns'
        ]
        
        importance = self.model.feature_importances_
        feature_importance = dict(zip(feature_names, importance))
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

def update_customer_churn_probabilities():
    """Update churn probabilities for all customers"""
    predictor = ChurnPredictor()
    customers = Customer.objects.all()
    
    for customer in customers:
        churn_probability = predictor.predict_churn(customer)
        customer.churn_probability = churn_probability
        customer.is_churned = churn_probability > 0.5  # Threshold for churn
        customer.save() 