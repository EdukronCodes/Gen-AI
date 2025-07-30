"""
Drug Toxicity Classification Model Validation
Handles cross-validation, hyperparameter tuning, and model performance assessment for molecular data
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
    train_test_split, GridSearchCV, RandomizedSearchCV,
    cross_val_predict
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    log_loss, cohen_kappa_score, matthews_corrcoef,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToxicityModelValidator:
    """Validator for drug toxicity classification models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_results = {}
        self.best_models = {}
        self.validation_history = []
        self.feature_importance_results = {}
        
    def validate_classification_model(self, model, X, y, model_name: str = "model", 
                                    feature_names: List[str] = None) -> Dict[str, Any]:
        """Validate classification model with comprehensive metrics"""
        try:
            results = {
                'model_name': model_name,
                'timestamp': datetime.now(),
                'metrics': {},
                'cross_validation': {},
                'confusion_matrix': None,
                'classification_report': None,
                'feature_importance': {},
                'roc_analysis': {},
                'pr_analysis': {}
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
            results['metrics']['cohen_kappa'] = cohen_kappa_score(y_test, y_pred)
            results['metrics']['matthews_corrcoef'] = matthews_corrcoef(y_test, y_pred)
            
            # ROC AUC if probabilities available
            if y_pred_proba is not None:
                results['metrics']['log_loss'] = log_loss(y_test, y_pred_proba)
                
                # ROC analysis
                if len(np.unique(y_test)) == 2:
                    results['metrics']['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                    results['metrics']['average_precision'] = average_precision_score(y_test, y_pred_proba[:, 1])
                    
                    # ROC curve data
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                    results['roc_analysis']['fpr'] = fpr.tolist()
                    results['roc_analysis']['tpr'] = tpr.tolist()
                    
                    # Precision-Recall curve data
                    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
                    results['pr_analysis']['precision'] = precision.tolist()
                    results['pr_analysis']['recall'] = recall.tolist()
            
            # Confusion matrix
            results['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
            
            # Classification report
            results['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            results['cross_validation']['mean_accuracy'] = cv_scores.mean()
            results['cross_validation']['std_accuracy'] = cv_scores.std()
            results['cross_validation']['cv_scores'] = cv_scores.tolist()
            
            # Feature importance
            if feature_names and hasattr(model, 'feature_importances_'):
                results['feature_importance']['tree_based'] = {
                    'feature_names': feature_names,
                    'importance_scores': model.feature_importances_.tolist()
                }
            
            # Store results
            self.validation_results[model_name] = results
            self.validation_history.append(results)
            
            logger.info(f"Validation completed for {model_name}")
            logger.info(f"Accuracy: {results['metrics']['accuracy']:.4f}")
            logger.info(f"F1 Score: {results['metrics']['f1_score']:.4f}")
            if 'roc_auc' in results['metrics']:
                logger.info(f"ROC AUC: {results['metrics']['roc_auc']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error validating classification model {model_name}: {e}")
            return {}
    
    def validate_neural_network(self, model, X, y, model_name: str = "neural_network") -> Dict[str, Any]:
        """Validate neural network model for toxicity classification"""
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
            
            epochs = 20
            train_losses = []
            val_losses = []
            
            for epoch in range(epochs):
                # Training
                model.train()
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                train_losses.append(epoch_loss / len(train_loader))
                
                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                val_losses.append(val_loss / len(test_loader))
            
            # Evaluation
            model.eval()
            y_pred = []
            y_true = []
            y_pred_proba = []
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    outputs = model(batch_X)
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                    y_pred.extend(predicted.numpy())
                    y_true.extend(batch_y.numpy())
                    y_pred_proba.extend(probabilities.numpy())
            
            # Metrics
            results['metrics']['accuracy'] = accuracy_score(y_true, y_pred)
            results['metrics']['precision'] = precision_score(y_true, y_pred, average='weighted')
            results['metrics']['recall'] = recall_score(y_true, y_pred, average='weighted')
            results['metrics']['f1_score'] = f1_score(y_true, y_pred, average='weighted')
            
            # ROC AUC for binary classification
            if len(np.unique(y_true)) == 2:
                y_pred_proba = np.array(y_pred_proba)
                results['metrics']['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                results['metrics']['average_precision'] = average_precision_score(y_true, y_pred_proba[:, 1])
            
            results['training_history']['train_losses'] = train_losses
            results['training_history']['val_losses'] = val_losses
            
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
                scoring='roc_auc' if len(np.unique(y)) == 2 else 'accuracy',
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
                scoring='roc_auc' if len(np.unique(y)) == 2 else 'accuracy',
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
    
    def compare_models(self, models: Dict[str, Any], X, y, feature_names: List[str] = None) -> Dict[str, Any]:
        """Compare multiple toxicity classification models"""
        try:
            comparison_results = {
                'timestamp': datetime.now(),
                'models': {},
                'summary': {},
                'rankings': {}
            }
            
            for model_name, model in models.items():
                logger.info(f"Validating model: {model_name}")
                
                if hasattr(model, 'predict_proba'):
                    # Classification model
                    results = self.validate_classification_model(model, X, y, model_name, feature_names)
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
                        'recall': results['metrics'].get('recall', 0),
                        'roc_auc': results['metrics'].get('roc_auc', 0)
                    }
            
            comparison_results['summary'] = summary
            
            # Find best model
            if summary:
                # Rank by ROC AUC if available, otherwise by accuracy
                if any('roc_auc' in model_metrics and model_metrics['roc_auc'] > 0 for model_metrics in summary.values()):
                    best_model = max(summary.items(), key=lambda x: x[1].get('roc_auc', 0))
                else:
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
    
    def validate_regression_model(self, model, X, y, model_name: str = "model") -> Dict[str, Any]:
        """Validate regression model (for toxicity prediction as continuous value)"""
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
            results['metrics']['mape'] = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            results['cross_validation']['mean_r2'] = cv_scores.mean()
            results['cross_validation']['std_r2'] = cv_scores.std()
            results['cross_validation']['cv_scores'] = cv_scores.tolist()
            
            # Store results
            self.validation_results[model_name] = results
            self.validation_history.append(results)
            
            logger.info(f"Regression validation completed for {model_name}")
            logger.info(f"R² Score: {results['metrics']['r2_score']:.4f}")
            logger.info(f"RMSE: {results['metrics']['rmse']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error validating regression model {model_name}: {e}")
            return {}
    
    def generate_validation_plots(self, model_name: str, output_dir: str = "plots") -> List[str]:
        """Generate validation visualization plots"""
        try:
            if model_name not in self.validation_results:
                logger.error(f"No validation results found for {model_name}")
                return []
            
            os.makedirs(output_dir, exist_ok=True)
            plot_files = []
            
            results = self.validation_results[model_name]
            
            # Confusion matrix plot
            if 'confusion_matrix' in results:
                plt.figure(figsize=(8, 6))
                cm = np.array(results['confusion_matrix'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix - {model_name}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                
                plot_path = os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files.append(plot_path)
            
            # ROC curve plot
            if 'roc_analysis' in results and results['roc_analysis']:
                plt.figure(figsize=(8, 6))
                fpr = results['roc_analysis']['fpr']
                tpr = results['roc_analysis']['tpr']
                plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {results["metrics"].get("roc_auc", 0):.3f})')
                plt.plot([0, 1], [0, 1], 'k--', label='Random')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {model_name}')
                plt.legend()
                
                plot_path = os.path.join(output_dir, f'{model_name}_roc_curve.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files.append(plot_path)
            
            # Precision-Recall curve plot
            if 'pr_analysis' in results and results['pr_analysis']:
                plt.figure(figsize=(8, 6))
                precision = results['pr_analysis']['precision']
                recall = results['pr_analysis']['recall']
                plt.plot(recall, precision, label=f'PR Curve (AP = {results["metrics"].get("average_precision", 0):.3f})')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'Precision-Recall Curve - {model_name}')
                plt.legend()
                
                plot_path = os.path.join(output_dir, f'{model_name}_pr_curve.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files.append(plot_path)
            
            # Feature importance plot
            if 'feature_importance' in results and results['feature_importance']:
                for imp_type, imp_data in results['feature_importance'].items():
                    if 'feature_names' in imp_data and 'importance_scores' in imp_data:
                        plt.figure(figsize=(12, 8))
                        importance_df = pd.DataFrame({
                            'feature': imp_data['feature_names'],
                            'importance': imp_data['importance_scores']
                        }).sort_values('importance', ascending=True).tail(20)
                        
                        plt.barh(range(len(importance_df)), importance_df['importance'])
                        plt.yticks(range(len(importance_df)), importance_df['feature'])
                        plt.title(f'Feature Importance ({imp_type}) - {model_name}')
                        plt.xlabel('Importance Score')
                        
                        plot_path = os.path.join(output_dir, f'{model_name}_{imp_type}_importance.png')
                        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        plot_files.append(plot_path)
            
            # Training history plot (for neural networks)
            if 'training_history' in results:
                plt.figure(figsize=(10, 6))
                train_losses = results['training_history'].get('train_losses', [])
                val_losses = results['training_history'].get('val_losses', [])
                
                epochs = range(1, len(train_losses) + 1)
                plt.plot(epochs, train_losses, 'b-', label='Training Loss')
                if val_losses:
                    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
                plt.title(f'Training History - {model_name}')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                
                plot_path = os.path.join(output_dir, f'{model_name}_training_history.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files.append(plot_path)
            
            logger.info(f"Generated {len(plot_files)} plots for {model_name}")
            return plot_files
            
        except Exception as e:
            logger.error(f"Error generating validation plots for {model_name}: {e}")
            return []
    
    def generate_validation_report(self, output_path: str = None) -> str:
        """Generate comprehensive validation report"""
        try:
            report = []
            report.append("=" * 80)
            report.append("DRUG TOXICITY CLASSIFICATION MODEL VALIDATION REPORT")
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
                    if 'roc_auc' in metrics:
                        report.append(f"ROC AUC: {metrics.get('roc_auc', 0):.4f}")
                    if 'average_precision' in metrics:
                        report.append(f"Average Precision: {metrics.get('average_precision', 0):.4f}")
                
                if 'cross_validation' in results:
                    cv = results['cross_validation']
                    if 'mean_accuracy' in cv:
                        report.append(f"CV Mean Accuracy: {cv.get('mean_accuracy', 0):.4f}")
                        report.append(f"CV Std Accuracy: {cv.get('std_accuracy', 0):.4f}")
                    elif 'mean_r2' in cv:
                        report.append(f"CV Mean R²: {cv.get('mean_r2', 0):.4f}")
                        report.append(f"CV Std R²: {cv.get('std_r2', 0):.4f}")
            
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
            best_roc_auc = 0
            best_accuracy = 0
            
            for model_name, results in self.validation_results.items():
                if 'metrics' in results:
                    metrics = results['metrics']
                    roc_auc = metrics.get('roc_auc', 0)
                    accuracy = metrics.get('accuracy', 0)
                    
                    if roc_auc > best_roc_auc:
                        best_roc_auc = roc_auc
                        best_model_name = model_name
                    elif accuracy > best_accuracy and roc_auc == 0:
                        best_accuracy = accuracy
                        best_model_name = model_name
            
            if best_model_name:
                report.append(f"Recommended model: {best_model_name}")
                if best_roc_auc > 0:
                    report.append(f"Best ROC AUC: {best_roc_auc:.4f}")
                else:
                    report.append(f"Best accuracy: {best_accuracy:.4f}")
            
            # Toxicity-specific recommendations
            report.append("\nToxicity Classification Specific Recommendations:")
            report.append("- Consider ensemble methods for improved robustness")
            report.append("- Focus on high recall to avoid missing toxic compounds")
            report.append("- Use domain-specific molecular descriptors")
            report.append("- Validate on external datasets when possible")
            
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
        'test_size': 0.2,
        'scoring_metric': 'roc_auc'
    }
    
    validator = ToxicityModelValidator(config)
    
    # Example data (replace with actual molecular data)
    X = np.random.rand(100, 50)  # Molecular descriptors
    y = np.random.randint(0, 2, 100)  # Binary toxicity labels
    
    # Example models
    models = {
        'random_forest': RandomForestClassifier(random_state=42),
        'logistic_regression': LogisticRegression(random_state=42),
        'svm': SVC(random_state=42, probability=True),
        'gradient_boosting': GradientBoostingClassifier(random_state=42)
    }
    
    # Validate models
    for model_name, model in models.items():
        validator.validate_classification_model(model, X, y, model_name)
    
    # Generate plots
    for model_name in models.keys():
        validator.generate_validation_plots(model_name)
    
    # Generate report
    report = validator.generate_validation_report("toxicity_validation_report.txt")
    print(report) 