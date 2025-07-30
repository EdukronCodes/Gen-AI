"""
Medical Chatbot Model Evaluation
Handles detailed performance analysis, error analysis, and model interpretability
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score,
    log_loss, cohen_kappa_score, matthews_corrcoef
)
from sklearn.model_selection import cross_val_predict
from sklearn.inspection import permutation_importance, partial_dependence
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import shap
import lime
import lime.lime_tabular
import lime.lime_text
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalModelEvaluator:
    """Comprehensive evaluator for medical chatbot models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.evaluation_results = {}
        self.error_analysis = {}
        self.interpretability_results = {}
        self.performance_history = []
        
    def evaluate_classification_model(self, model, X, y, X_test=None, y_test=None, 
                                    model_name: str = "model") -> Dict[str, Any]:
        """Comprehensive evaluation of classification model"""
        try:
            results = {
                'model_name': model_name,
                'timestamp': datetime.now(),
                'metrics': {},
                'detailed_analysis': {},
                'error_analysis': {},
                'interpretability': {}
            }
            
            # Use provided test set or split data
            if X_test is None or y_test is None:
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                model.fit(X_train, y_train)
            else:
                X_train, y_train = X, y
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Basic metrics
            results['metrics']['accuracy'] = accuracy_score(y_test, y_pred)
            results['metrics']['precision_macro'] = precision_score(y_test, y_pred, average='macro')
            results['metrics']['precision_weighted'] = precision_score(y_test, y_pred, average='weighted')
            results['metrics']['recall_macro'] = recall_score(y_test, y_pred, average='macro')
            results['metrics']['recall_weighted'] = recall_score(y_test, y_pred, average='weighted')
            results['metrics']['f1_macro'] = f1_score(y_test, y_pred, average='macro')
            results['metrics']['f1_weighted'] = f1_score(y_test, y_pred, average='weighted')
            
            # Additional metrics
            results['metrics']['cohen_kappa'] = cohen_kappa_score(y_test, y_pred)
            results['metrics']['matthews_corrcoef'] = matthews_corrcoef(y_test, y_pred)
            
            # Probability-based metrics
            if y_pred_proba is not None:
                results['metrics']['log_loss'] = log_loss(y_test, y_pred_proba)
                
                # ROC AUC for binary classification
                if len(np.unique(y_test)) == 2:
                    results['metrics']['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                    results['metrics']['average_precision'] = average_precision_score(y_test, y_pred_proba[:, 1])
            
            # Detailed analysis
            results['detailed_analysis']['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
            results['detailed_analysis']['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
            
            # Error analysis
            results['error_analysis'] = self.analyze_classification_errors(X_test, y_test, y_pred, y_pred_proba)
            
            # Store results
            self.evaluation_results[model_name] = results
            self.performance_history.append(results)
            
            logger.info(f"Evaluation completed for {model_name}")
            logger.info(f"Accuracy: {results['metrics']['accuracy']:.4f}")
            logger.info(f"F1 Score (Weighted): {results['metrics']['f1_weighted']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating classification model {model_name}: {e}")
            return {}
    
    def evaluate_regression_model(self, model, X, y, X_test=None, y_test=None, 
                                model_name: str = "model") -> Dict[str, Any]:
        """Comprehensive evaluation of regression model"""
        try:
            results = {
                'model_name': model_name,
                'timestamp': datetime.now(),
                'metrics': {},
                'detailed_analysis': {},
                'error_analysis': {}
            }
            
            # Use provided test set or split data
            if X_test is None or y_test is None:
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                model.fit(X_train, y_train)
            else:
                X_train, y_train = X, y
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            results['metrics']['mse'] = mean_squared_error(y_test, y_pred)
            results['metrics']['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
            results['metrics']['mae'] = mean_absolute_error(y_test, y_pred)
            results['metrics']['r2_score'] = r2_score(y_test, y_pred)
            results['metrics']['mape'] = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            # Additional metrics
            results['metrics']['explained_variance'] = 1 - np.var(y_test - y_pred) / np.var(y_test)
            
            # Error analysis
            results['error_analysis'] = self.analyze_regression_errors(y_test, y_pred)
            
            # Store results
            self.evaluation_results[model_name] = results
            self.performance_history.append(results)
            
            logger.info(f"Evaluation completed for {model_name}")
            logger.info(f"R² Score: {results['metrics']['r2_score']:.4f}")
            logger.info(f"RMSE: {results['metrics']['rmse']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating regression model {model_name}: {e}")
            return {}
    
    def analyze_classification_errors(self, X_test, y_test, y_pred, y_pred_proba=None) -> Dict[str, Any]:
        """Analyze classification errors in detail"""
        try:
            error_analysis = {
                'error_indices': [],
                'error_types': {},
                'confidence_analysis': {},
                'feature_importance_errors': {}
            }
            
            # Find error indices
            error_mask = y_test != y_pred
            error_indices = np.where(error_mask)[0]
            error_analysis['error_indices'] = error_indices.tolist()
            
            # Analyze error types
            error_types = {}
            for idx in error_indices:
                true_label = y_test[idx]
                pred_label = y_pred[idx]
                error_key = f"{true_label}_to_{pred_label}"
                error_types[error_key] = error_types.get(error_key, 0) + 1
            
            error_analysis['error_types'] = error_types
            
            # Confidence analysis for probabilistic models
            if y_pred_proba is not None:
                confidence_scores = np.max(y_pred_proba, axis=1)
                error_analysis['confidence_analysis'] = {
                    'correct_confidence_mean': np.mean(confidence_scores[~error_mask]),
                    'error_confidence_mean': np.mean(confidence_scores[error_mask]),
                    'confidence_threshold_analysis': self.analyze_confidence_thresholds(
                        confidence_scores, error_mask
                    )
                }
            
            return error_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing classification errors: {e}")
            return {}
    
    def analyze_regression_errors(self, y_test, y_pred) -> Dict[str, Any]:
        """Analyze regression errors in detail"""
        try:
            errors = y_test - y_pred
            abs_errors = np.abs(errors)
            
            error_analysis = {
                'error_statistics': {
                    'mean_error': np.mean(errors),
                    'std_error': np.std(errors),
                    'mean_abs_error': np.mean(abs_errors),
                    'median_abs_error': np.median(abs_errors),
                    'max_error': np.max(abs_errors),
                    'min_error': np.min(abs_errors)
                },
                'error_distribution': {
                    'percentiles': np.percentile(abs_errors, [25, 50, 75, 90, 95, 99]).tolist()
                },
                'outlier_analysis': {
                    'outlier_threshold': np.percentile(abs_errors, 95),
                    'outlier_count': np.sum(abs_errors > np.percentile(abs_errors, 95))
                }
            }
            
            return error_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing regression errors: {e}")
            return {}
    
    def analyze_confidence_thresholds(self, confidence_scores, error_mask) -> Dict[str, Any]:
        """Analyze model performance at different confidence thresholds"""
        try:
            thresholds = np.arange(0.5, 1.0, 0.05)
            threshold_analysis = {}
            
            for threshold in thresholds:
                high_conf_mask = confidence_scores >= threshold
                if np.sum(high_conf_mask) > 0:
                    high_conf_accuracy = 1 - np.mean(error_mask[high_conf_mask])
                    threshold_analysis[f"threshold_{threshold:.2f}"] = {
                        'accuracy': high_conf_accuracy,
                        'coverage': np.mean(high_conf_mask)
                    }
            
            return threshold_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing confidence thresholds: {e}")
            return {}
    
    def perform_model_interpretability(self, model, X, feature_names=None, 
                                     model_name: str = "model") -> Dict[str, Any]:
        """Perform model interpretability analysis"""
        try:
            interpretability_results = {
                'model_name': model_name,
                'timestamp': datetime.now(),
                'feature_importance': {},
                'partial_dependence': {},
                'shap_analysis': {},
                'lime_analysis': {}
            }
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                interpretability_results['feature_importance']['tree_based'] = {
                    'importance_scores': model.feature_importances_.tolist(),
                    'feature_names': feature_names or [f"feature_{i}" for i in range(len(model.feature_importances_))]
                }
            
            # Permutation importance
            try:
                perm_importance = permutation_importance(model, X, np.random.randint(0, 2, len(X)), 
                                                       n_repeats=10, random_state=42)
                interpretability_results['feature_importance']['permutation'] = {
                    'importance_scores': perm_importance.importances_mean.tolist(),
                    'importance_std': perm_importance.importances_std.tolist(),
                    'feature_names': feature_names or [f"feature_{i}" for i in range(len(perm_importance.importances_mean))]
                }
            except Exception as e:
                logger.warning(f"Could not compute permutation importance: {e}")
            
            # SHAP analysis for tree-based models
            if isinstance(model, RandomForestClassifier):
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X[:100])  # Use subset for efficiency
                    
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # For binary classification
                    
                    interpretability_results['shap_analysis'] = {
                        'shap_values': shap_values.tolist(),
                        'expected_value': explainer.expected_value
                    }
                except Exception as e:
                    logger.warning(f"Could not compute SHAP values: {e}")
            
            # LIME analysis
            try:
                if feature_names:
                    explainer = lime.lime_tabular.LimeTabularExplainer(
                        X, feature_names=feature_names, class_names=['class_0', 'class_1']
                    )
                    lime_explanation = explainer.explain_instance(
                        X[0], model.predict_proba, num_features=10
                    )
                    interpretability_results['lime_analysis']['sample_explanation'] = {
                        'feature_weights': dict(lime_explanation.as_list()),
                        'prediction': lime_explanation.predicted_value
                    }
            except Exception as e:
                logger.warning(f"Could not compute LIME explanation: {e}")
            
            # Store results
            self.interpretability_results[model_name] = interpretability_results
            
            return interpretability_results
            
        except Exception as e:
            logger.error(f"Error performing model interpretability for {model_name}: {e}")
            return {}
    
    def generate_performance_plots(self, model_name: str, output_dir: str = "plots") -> List[str]:
        """Generate performance visualization plots"""
        try:
            if model_name not in self.evaluation_results:
                logger.error(f"No evaluation results found for {model_name}")
                return []
            
            os.makedirs(output_dir, exist_ok=True)
            plot_files = []
            
            results = self.evaluation_results[model_name]
            
            # Confusion matrix plot
            if 'confusion_matrix' in results.get('detailed_analysis', {}):
                plt.figure(figsize=(8, 6))
                cm = np.array(results['detailed_analysis']['confusion_matrix'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix - {model_name}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                
                plot_path = os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files.append(plot_path)
            
            # Feature importance plot
            if model_name in self.interpretability_results:
                interpretability = self.interpretability_results[model_name]
                if 'feature_importance' in interpretability:
                    for imp_type, imp_data in interpretability['feature_importance'].items():
                        if 'importance_scores' in imp_data and 'feature_names' in imp_data:
                            plt.figure(figsize=(10, 6))
                            importance_df = pd.DataFrame({
                                'feature': imp_data['feature_names'],
                                'importance': imp_data['importance_scores']
                            }).sort_values('importance', ascending=True)
                            
                            plt.barh(range(len(importance_df)), importance_df['importance'])
                            plt.yticks(range(len(importance_df)), importance_df['feature'])
                            plt.title(f'Feature Importance ({imp_type}) - {model_name}')
                            plt.xlabel('Importance Score')
                            
                            plot_path = os.path.join(output_dir, f'{model_name}_{imp_type}_importance.png')
                            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                            plt.close()
                            plot_files.append(plot_path)
            
            # Error analysis plots
            if 'error_analysis' in results:
                error_analysis = results['error_analysis']
                
                # Error type distribution
                if 'error_types' in error_analysis:
                    plt.figure(figsize=(10, 6))
                    error_types = error_analysis['error_types']
                    plt.bar(range(len(error_types)), list(error_types.values()))
                    plt.xticks(range(len(error_types)), list(error_types.keys()), rotation=45)
                    plt.title(f'Error Type Distribution - {model_name}')
                    plt.ylabel('Count')
                    
                    plot_path = os.path.join(output_dir, f'{model_name}_error_types.png')
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    plot_files.append(plot_path)
            
            logger.info(f"Generated {len(plot_files)} plots for {model_name}")
            return plot_files
            
        except Exception as e:
            logger.error(f"Error generating performance plots for {model_name}: {e}")
            return []
    
    def generate_evaluation_report(self, model_name: str, output_path: str = None) -> str:
        """Generate comprehensive evaluation report"""
        try:
            if model_name not in self.evaluation_results:
                return f"No evaluation results found for {model_name}"
            
            results = self.evaluation_results[model_name]
            
            report = []
            report.append("=" * 80)
            report.append(f"MODEL EVALUATION REPORT - {model_name.upper()}")
            report.append("=" * 80)
            report.append(f"Generated: {results['timestamp']}")
            report.append("")
            
            # Metrics summary
            report.append("PERFORMANCE METRICS")
            report.append("-" * 40)
            
            metrics = results['metrics']
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    report.append(f"{metric_name.replace('_', ' ').title()}: {value:.4f}")
                else:
                    report.append(f"{metric_name.replace('_', ' ').title()}: {value}")
            
            # Error analysis
            if 'error_analysis' in results:
                report.append("\n\nERROR ANALYSIS")
                report.append("-" * 40)
                
                error_analysis = results['error_analysis']
                if 'error_types' in error_analysis:
                    report.append("Error Type Distribution:")
                    for error_type, count in error_analysis['error_types'].items():
                        report.append(f"  {error_type}: {count}")
                
                if 'confidence_analysis' in error_analysis:
                    conf_analysis = error_analysis['confidence_analysis']
                    report.append(f"\nConfidence Analysis:")
                    report.append(f"  Correct predictions confidence: {conf_analysis.get('correct_confidence_mean', 0):.4f}")
                    report.append(f"  Error predictions confidence: {conf_analysis.get('error_confidence_mean', 0):.4f}")
            
            # Interpretability summary
            if model_name in self.interpretability_results:
                report.append("\n\nINTERPRETABILITY ANALYSIS")
                report.append("-" * 40)
                
                interpretability = self.interpretability_results[model_name]
                if 'feature_importance' in interpretability:
                    report.append("Feature Importance Analysis Available")
                    for imp_type in interpretability['feature_importance'].keys():
                        report.append(f"  - {imp_type} importance computed")
            
            # Recommendations
            report.append("\n\nRECOMMENDATIONS")
            report.append("-" * 40)
            
            # Performance-based recommendations
            accuracy = metrics.get('accuracy', 0)
            f1_score = metrics.get('f1_weighted', 0)
            
            if accuracy < 0.7:
                report.append("- Model accuracy is below 70%. Consider:")
                report.append("  * Collecting more training data")
                report.append("  * Feature engineering")
                report.append("  * Hyperparameter tuning")
            
            if f1_score < 0.6:
                report.append("- F1 score indicates class imbalance issues. Consider:")
                report.append("  * Class balancing techniques")
                report.append("  * Different evaluation metrics")
                report.append("  * Ensemble methods")
            
            if 'error_analysis' in results and 'error_types' in results['error_analysis']:
                error_types = results['error_analysis']['error_types']
                if len(error_types) > 3:
                    report.append("- Multiple error types detected. Consider:")
                    report.append("  * Analyzing specific error patterns")
                    report.append("  * Domain-specific feature engineering")
                    report.append("  * Model ensemble approaches")
            
            report_text = "\n".join(report)
            
            # Save report
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(report_text)
                logger.info(f"Evaluation report saved to {output_path}")
            
            return report_text
            
        except Exception as e:
            logger.error(f"Error generating evaluation report for {model_name}: {e}")
            return f"Error generating evaluation report for {model_name}"
    
    def compare_model_performance(self, model_names: List[str]) -> Dict[str, Any]:
        """Compare performance across multiple models"""
        try:
            comparison = {
                'timestamp': datetime.now(),
                'models': {},
                'summary': {},
                'rankings': {}
            }
            
            # Collect metrics for each model
            for model_name in model_names:
                if model_name in self.evaluation_results:
                    results = self.evaluation_results[model_name]
                    comparison['models'][model_name] = results['metrics']
            
            # Create summary
            if comparison['models']:
                metrics_to_compare = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
                
                for metric in metrics_to_compare:
                    metric_values = {}
                    for model_name, metrics in comparison['models'].items():
                        if metric in metrics:
                            metric_values[model_name] = metrics[metric]
                    
                    if metric_values:
                        # Rank models by this metric
                        sorted_models = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
                        comparison['rankings'][metric] = sorted_models
                        
                        comparison['summary'][metric] = {
                            'best_model': sorted_models[0][0],
                            'best_score': sorted_models[0][1],
                            'worst_model': sorted_models[-1][0],
                            'worst_score': sorted_models[-1][1],
                            'mean_score': np.mean(list(metric_values.values())),
                            'std_score': np.std(list(metric_values.values()))
                        }
            
            logger.info("Model performance comparison completed")
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing model performance: {e}")
            return {}
    
    def save_evaluation_results(self, filepath: str):
        """Save evaluation results to file"""
        try:
            # Convert results to serializable format
            serializable_results = {}
            for model_name, results in self.evaluation_results.items():
                serializable_results[model_name] = {
                    'model_name': results.get('model_name'),
                    'timestamp': str(results.get('timestamp')),
                    'metrics': results.get('metrics', {}),
                    'detailed_analysis': results.get('detailed_analysis', {}),
                    'error_analysis': results.get('error_analysis', {})
                }
            
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Evaluation results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
    
    def load_evaluation_results(self, filepath: str):
        """Load evaluation results from file"""
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            # Convert back to proper format
            for model_name, result in results.items():
                if 'timestamp' in result:
                    result['timestamp'] = datetime.fromisoformat(result['timestamp'])
            
            self.evaluation_results = results
            logger.info(f"Evaluation results loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading evaluation results: {e}")

# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'evaluation_metrics': ['accuracy', 'f1_score', 'precision', 'recall'],
        'confidence_thresholds': [0.5, 0.7, 0.9],
        'plot_output_dir': 'evaluation_plots'
    }
    
    evaluator = MedicalModelEvaluator(config)
    
    # Example data (replace with actual data)
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 3, 100)
    
    # Example model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    
    # Evaluate model
    results = evaluator.evaluate_classification_model(model, X, y, model_name="random_forest")
    
    # Generate plots
    plot_files = evaluator.generate_performance_plots("random_forest")
    
    # Generate report
    report = evaluator.generate_evaluation_report("random_forest", "evaluation_report.txt")
    print(report) 