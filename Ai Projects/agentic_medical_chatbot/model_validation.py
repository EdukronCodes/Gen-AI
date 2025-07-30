"""
Medical Chatbot Model Validation
Handles cross-validation, hyperparameter tuning, and model performance assessment
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, KFold, 
    train_test_split, GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalModelValidator:
    """Validator for medical chatbot models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_results = {}
        self.best_models = {}
        self.validation_history = []
        
    def validate_classification_model(self, model, X, y, model_name: str = "model") -> Dict[str, Any]:
        """Validate classification model with comprehensive metrics"""
        try:
            results = {
                'model_name': model_name,
                'timestamp': datetime.now(),
                'metrics': {},
                'cross_validation': {},
                'confusion_matrix': None,
                'classification_report': None
            }
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Basic metrics
            results['metrics']['accuracy'] = accuracy_score(y_test, y_pred)
            results['metrics']['precision'] = precision_score(y_test, y_pred, average='weighted')
            results['metrics']['recall'] = recall_score(y_test, y_pred, average='weighted')
            results['metrics']['f1_score'] = f1_score(y_test, y_pred, average='weighted')
            
            # ROC AUC if probabilities available
            if y_pred_proba is not None and len(np.unique(y)) == 2:
                results['metrics']['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            
            # Confusion matrix
            results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
            
            # Classification report
            results['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            results['cross_validation']['mean_accuracy'] = cv_scores.mean()
            results['cross_validation']['std_accuracy'] = cv_scores.std()
            results['cross_validation']['cv_scores'] = cv_scores.tolist()
            
            # Store results
            self.validation_results[model_name] = results
            self.validation_history.append(results)
            
            logger.info(f"Validation completed for {model_name}")
            logger.info(f"Accuracy: {results['metrics']['accuracy']:.4f}")
            logger.info(f"F1 Score: {results['metrics']['f1_score']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error validating classification model {model_name}: {e}")
            return {}
    
    def validate_regression_model(self, model, X, y, model_name: str = "model") -> Dict[str, Any]:
        """Validate regression model with comprehensive metrics"""
        try:
            results = {
                'model_name': model_name,
                'timestamp': datetime.now(),
                'metrics': {},
                'cross_validation': {}
            }
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            results['metrics']['mse'] = mean_squared_error(y_test, y_pred)
            results['metrics']['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
            results['metrics']['mae'] = mean_absolute_error(y_test, y_pred)
            results['metrics']['r2_score'] = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            results['cross_validation']['mean_r2'] = cv_scores.mean()
            results['cross_validation']['std_r2'] = cv_scores.std()
            results['cross_validation']['cv_scores'] = cv_scores.tolist()
            
            # Store results
            self.validation_results[model_name] = results
            self.validation_history.append(results)
            
            logger.info(f"Validation completed for {model_name}")
            logger.info(f"R² Score: {results['metrics']['r2_score']:.4f}")
            logger.info(f"RMSE: {results['metrics']['rmse']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error validating regression model {model_name}: {e}")
            return {}
    
    def validate_neural_network(self, model, X, y, model_name: str = "neural_network") -> Dict[str, Any]:
        """Validate neural network model"""
        try:
            results = {
                'model_name': model_name,
                'timestamp': datetime.now(),
                'metrics': {},
                'training_history': {}
            }
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.LongTensor(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor
            )
            
            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train)
            test_dataset = TensorDataset(X_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            # Training
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            epochs = 10
            train_losses = []
            
            for epoch in range(epochs):
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                train_losses.append(epoch_loss / len(train_loader))
            
            # Evaluation
            model.eval()
            y_pred = []
            y_true = []
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    outputs = model(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                    y_pred.extend(predicted.numpy())
                    y_true.extend(batch_y.numpy())
            
            # Metrics
            results['metrics']['accuracy'] = accuracy_score(y_true, y_pred)
            results['metrics']['precision'] = precision_score(y_true, y_pred, average='weighted')
            results['metrics']['recall'] = recall_score(y_true, y_pred, average='weighted')
            results['metrics']['f1_score'] = f1_score(y_true, y_pred, average='weighted')
            
            results['training_history']['train_losses'] = train_losses
            
            # Store results
            self.validation_results[model_name] = results
            self.validation_history.append(results)
            
            logger.info(f"Neural network validation completed for {model_name}")
            logger.info(f"Accuracy: {results['metrics']['accuracy']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error validating neural network {model_name}: {e}")
            return {}
    
    def perform_hyperparameter_tuning(self, model_class, X, y, param_grid: Dict[str, List], 
                                    model_name: str = "model", cv: int = 5) -> Dict[str, Any]:
        """Perform hyperparameter tuning using GridSearchCV"""
        try:
            logger.info(f"Starting hyperparameter tuning for {model_name}")
            
            # Grid search
            grid_search = GridSearchCV(
                estimator=model_class(),
                param_grid=param_grid,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X, y)
            
            results = {
                'model_name': model_name,
                'timestamp': datetime.now(),
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'best_estimator': grid_search.best_estimator_,
                'cv_results': grid_search.cv_results_
            }
            
            # Validate best model
            best_model_results = self.validate_classification_model(
                grid_search.best_estimator_, X, y, f"{model_name}_best"
            )
            results['validation_results'] = best_model_results
            
            # Store best model
            self.best_models[model_name] = grid_search.best_estimator_
            
            logger.info(f"Hyperparameter tuning completed for {model_name}")
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning for {model_name}: {e}")
            return {}
    
    def perform_randomized_search(self, model_class, X, y, param_distributions: Dict[str, List], 
                                model_name: str = "model", n_iter: int = 100, cv: int = 5) -> Dict[str, Any]:
        """Perform hyperparameter tuning using RandomizedSearchCV"""
        try:
            logger.info(f"Starting randomized search for {model_name}")
            
            # Random search
            random_search = RandomizedSearchCV(
                estimator=model_class(),
                param_distributions=param_distributions,
                n_iter=n_iter,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
            
            random_search.fit(X, y)
            
            results = {
                'model_name': model_name,
                'timestamp': datetime.now(),
                'best_params': random_search.best_params_,
                'best_score': random_search.best_score_,
                'best_estimator': random_search.best_estimator_,
                'cv_results': random_search.cv_results_
            }
            
            # Validate best model
            best_model_results = self.validate_classification_model(
                random_search.best_estimator_, X, y, f"{model_name}_best"
            )
            results['validation_results'] = best_model_results
            
            # Store best model
            self.best_models[model_name] = random_search.best_estimator_
            
            logger.info(f"Randomized search completed for {model_name}")
            logger.info(f"Best parameters: {random_search.best_params_}")
            logger.info(f"Best CV score: {random_search.best_score_:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in randomized search for {model_name}: {e}")
            return {}
    
    def compare_models(self, models: Dict[str, Any], X, y) -> Dict[str, Any]:
        """Compare multiple models"""
        try:
            comparison_results = {
                'timestamp': datetime.now(),
                'models': {},
                'summary': {}
            }
            
            for model_name, model in models.items():
                logger.info(f"Validating model: {model_name}")
                
                if hasattr(model, 'predict_proba'):
                    # Classification model
                    results = self.validate_classification_model(model, X, y, model_name)
                else:
                    # Regression model
                    results = self.validate_regression_model(model, X, y, model_name)
                
                comparison_results['models'][model_name] = results
            
            # Create summary
            summary = {}
            for model_name, results in comparison_results['models'].items():
                if 'metrics' in results:
                    summary[model_name] = {
                        'accuracy': results['metrics'].get('accuracy', 0),
                        'f1_score': results['metrics'].get('f1_score', 0),
                        'precision': results['metrics'].get('precision', 0),
                        'recall': results['metrics'].get('recall', 0)
                    }
            
            comparison_results['summary'] = summary
            
            # Find best model
            if summary:
                best_model = max(summary.items(), key=lambda x: x[1]['accuracy'])
                comparison_results['best_model'] = {
                    'name': best_model[0],
                    'metrics': best_model[1]
                }
            
            logger.info("Model comparison completed")
            logger.info(f"Best model: {comparison_results.get('best_model', {}).get('name', 'None')}")
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return {}
    
    def generate_validation_report(self, output_path: str = None) -> str:
        """Generate comprehensive validation report"""
        try:
            report = []
            report.append("=" * 80)
            report.append("MEDICAL CHATBOT MODEL VALIDATION REPORT")
            report.append("=" * 80)
            report.append(f"Generated: {datetime.now()}")
            report.append("")
            
            # Summary of all validations
            report.append("VALIDATION SUMMARY")
            report.append("-" * 40)
            
            for model_name, results in self.validation_results.items():
                report.append(f"\nModel: {model_name}")
                report.append(f"Timestamp: {results.get('timestamp', 'N/A')}")
                
                if 'metrics' in results:
                    metrics = results['metrics']
                    report.append(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
                    report.append(f"Precision: {metrics.get('precision', 0):.4f}")
                    report.append(f"Recall: {metrics.get('recall', 0):.4f}")
                    report.append(f"F1 Score: {metrics.get('f1_score', 0):.4f}")
                
                if 'cross_validation' in results:
                    cv = results['cross_validation']
                    report.append(f"CV Mean Accuracy: {cv.get('mean_accuracy', 0):.4f}")
                    report.append(f"CV Std Accuracy: {cv.get('std_accuracy', 0):.4f}")
            
            # Best models
            if self.best_models:
                report.append("\n\nBEST MODELS")
                report.append("-" * 40)
                for model_name, model in self.best_models.items():
                    report.append(f"{model_name}: {type(model).__name__}")
            
            # Recommendations
            report.append("\n\nRECOMMENDATIONS")
            report.append("-" * 40)
            
            # Find best performing model
            best_model_name = None
            best_accuracy = 0
            
            for model_name, results in self.validation_results.items():
                if 'metrics' in results:
                    accuracy = results['metrics'].get('accuracy', 0)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model_name = model_name
            
            if best_model_name:
                report.append(f"Recommended model: {best_model_name}")
                report.append(f"Best accuracy: {best_accuracy:.4f}")
            
            report_text = "\n".join(report)
            
            # Save report
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(report_text)
                logger.info(f"Validation report saved to {output_path}")
            
            return report_text
            
        except Exception as e:
            logger.error(f"Error generating validation report: {e}")
            return "Error generating validation report"
    
    def save_validation_results(self, filepath: str):
        """Save validation results to file"""
        try:
            # Convert results to serializable format
            serializable_results = {}
            for model_name, results in self.validation_results.items():
                serializable_results[model_name] = {
                    'model_name': results.get('model_name'),
                    'timestamp': str(results.get('timestamp')),
                    'metrics': results.get('metrics', {}),
                    'cross_validation': results.get('cross_validation', {})
                }
            
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Validation results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving validation results: {e}")
    
    def load_validation_results(self, filepath: str):
        """Load validation results from file"""
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            # Convert back to proper format
            for model_name, result in results.items():
                if 'timestamp' in result:
                    result['timestamp'] = datetime.fromisoformat(result['timestamp'])
            
            self.validation_results = results
            logger.info(f"Validation results loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading validation results: {e}")

# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'validation_cv_folds': 5,
        'random_state': 42,
        'test_size': 0.2
    }
    
    validator = MedicalModelValidator(config)
    
    # Example data (replace with actual data)
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 3, 100)
    
    # Example models
    models = {
        'random_forest': RandomForestClassifier(random_state=42),
        'logistic_regression': LogisticRegression(random_state=42),
        'svm': SVC(random_state=42, probability=True)
    }
    
    # Validate models
    for model_name, model in models.items():
        validator.validate_classification_model(model, X, y, model_name)
    
    # Generate report
    report = validator.generate_validation_report("validation_report.txt")
    print(report) 