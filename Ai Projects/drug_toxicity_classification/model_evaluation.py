"""
Drug Toxicity Classification Model Evaluation
Handles detailed performance analysis, error analysis, and model interpretability for molecular data
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
    log_loss, cohen_kappa_score, matthews_corrcoef,
    brier_score_loss, calibration_curve
)
from sklearn.model_selection import cross_val_predict
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import shap
import lime
import lime.lime_tabular
import lime.lime_text
import joblib

# Chemical informatics imports
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToxicityModelEvaluator:
    """Comprehensive evaluator for drug toxicity classification models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.evaluation_results = {}
        self.error_analysis = {}
        self.interpretability_results = {}
        self.performance_history = []
        self.molecular_insights = {}
        
    def evaluate_classification_model(self, model, X, y, X_test=None, y_test=None, 
                                    model_name: str = "model", feature_names: List[str] = None,
                                    smiles_data: List[str] = None) -> Dict[str, Any]:
        """Comprehensive evaluation of toxicity classification model"""
        try:
            results = {
                'model_name': model_name,
                'timestamp': datetime.now(),
                'metrics': {},
                'detailed_analysis': {},
                'error_analysis': {},
                'interpretability': {},
                'molecular_insights': {}
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
                results['metrics']['brier_score'] = brier_score_loss(y_test, y_pred_proba[:, 1])
                
                # ROC AUC for binary classification
                if len(np.unique(y_test)) == 2:
                    results['metrics']['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                    results['metrics']['average_precision'] = average_precision_score(y_test, y_pred_proba[:, 1])
            
            # Detailed analysis
            results['detailed_analysis']['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
            results['detailed_analysis']['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
            
            # Error analysis
            results['error_analysis'] = self.analyze_toxicity_errors(X_test, y_test, y_pred, y_pred_proba, smiles_data)
            
            # Interpretability analysis
            if feature_names:
                results['interpretability'] = self.perform_toxicity_interpretability(
                    model, X_test, feature_names, y_test, y_pred
                )
            
            # Molecular insights
            if smiles_data is not None:
                results['molecular_insights'] = self.analyze_molecular_patterns(
                    smiles_data, y_test, y_pred, y_pred_proba
                )
            
            # Store results
            self.evaluation_results[model_name] = results
            self.performance_history.append(results)
            
            logger.info(f"Evaluation completed for {model_name}")
            logger.info(f"Accuracy: {results['metrics']['accuracy']:.4f}")
            logger.info(f"F1 Score (Weighted): {results['metrics']['f1_weighted']:.4f}")
            if 'roc_auc' in results['metrics']:
                logger.info(f"ROC AUC: {results['metrics']['roc_auc']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating classification model {model_name}: {e}")
            return {}
    
    def analyze_toxicity_errors(self, X_test, y_test, y_pred, y_pred_proba=None, 
                               smiles_data: List[str] = None) -> Dict[str, Any]:
        """Analyze toxicity classification errors in detail"""
        try:
            error_analysis = {
                'error_indices': [],
                'error_types': {},
                'confidence_analysis': {},
                'molecular_error_patterns': {},
                'toxicity_threshold_analysis': {}
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
                
                # Toxicity threshold analysis
                if len(np.unique(y_test)) == 2:
                    error_analysis['toxicity_threshold_analysis'] = self.analyze_toxicity_thresholds(
                        y_pred_proba[:, 1], y_test, error_mask
                    )
            
            # Molecular error patterns
            if smiles_data is not None:
                error_analysis['molecular_error_patterns'] = self.analyze_molecular_error_patterns(
                    smiles_data, error_indices, y_test, y_pred
                )
            
            return error_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing toxicity errors: {e}")
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
                        'coverage': np.mean(high_conf_mask),
                        'error_rate': np.mean(error_mask[high_conf_mask])
                    }
            
            return threshold_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing confidence thresholds: {e}")
            return {}
    
    def analyze_toxicity_thresholds(self, toxicity_probs, y_true, error_mask) -> Dict[str, Any]:
        """Analyze toxicity prediction at different probability thresholds"""
        try:
            thresholds = np.arange(0.1, 1.0, 0.1)
            threshold_analysis = {}
            
            for threshold in thresholds:
                pred_toxic = toxicity_probs >= threshold
                accuracy = accuracy_score(y_true, pred_toxic)
                precision = precision_score(y_true, pred_toxic, zero_division=0)
                recall = recall_score(y_true, pred_toxic, zero_division=0)
                f1 = f1_score(y_true, pred_toxic, zero_division=0)
                
                threshold_analysis[f"threshold_{threshold:.1f}"] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'predicted_toxic_rate': np.mean(pred_toxic)
                }
            
            return threshold_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing toxicity thresholds: {e}")
            return {}
    
    def analyze_molecular_error_patterns(self, smiles_data: List[str], error_indices: List[int], 
                                       y_true, y_pred) -> Dict[str, Any]:
        """Analyze molecular patterns in prediction errors"""
        try:
            error_patterns = {
                'error_molecules': [],
                'molecular_properties': {},
                'structural_patterns': {}
            }
            
            # Get error molecules
            error_smiles = [smiles_data[i] for i in error_indices if i < len(smiles_data)]
            error_patterns['error_molecules'] = error_smiles
            
            # Analyze molecular properties of errors
            if error_smiles:
                mol_properties = []
                for smiles in error_smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        properties = {
                            'mol_weight': Descriptors.MolWt(mol),
                            'logp': Descriptors.MolLogP(mol),
                            'num_h_donors': Descriptors.NumHDonors(mol),
                            'num_h_acceptors': Descriptors.NumHAcceptors(mol),
                            'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                            'tpsa': Descriptors.TPSA(mol)
                        }
                        mol_properties.append(properties)
                
                if mol_properties:
                    # Calculate statistics
                    error_patterns['molecular_properties'] = {
                        'mean_mol_weight': np.mean([p['mol_weight'] for p in mol_properties]),
                        'mean_logp': np.mean([p['logp'] for p in mol_properties]),
                        'mean_h_donors': np.mean([p['num_h_donors'] for p in mol_properties]),
                        'mean_h_acceptors': np.mean([p['num_h_acceptors'] for p in mol_properties]),
                        'mean_rotatable_bonds': np.mean([p['num_rotatable_bonds'] for p in mol_properties]),
                        'mean_tpsa': np.mean([p['tpsa'] for p in mol_properties])
                    }
            
            return error_patterns
            
        except Exception as e:
            logger.error(f"Error analyzing molecular error patterns: {e}")
            return {}
    
    def analyze_molecular_patterns(self, smiles_data: List[str], y_true, y_pred, 
                                 y_pred_proba=None) -> Dict[str, Any]:
        """Analyze molecular patterns in predictions"""
        try:
            molecular_insights = {
                'toxicity_distribution': {},
                'molecular_property_correlations': {},
                'structural_insights': {}
            }
            
            # Toxicity distribution analysis
            toxicity_counts = {}
            for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
                key = f"true_{true_label}_pred_{pred_label}"
                toxicity_counts[key] = toxicity_counts.get(key, 0) + 1
            
            molecular_insights['toxicity_distribution'] = toxicity_counts
            
            # Molecular property correlations
            if len(smiles_data) == len(y_true):
                mol_properties = []
                for smiles in smiles_data:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        properties = {
                            'mol_weight': Descriptors.MolWt(mol),
                            'logp': Descriptors.MolLogP(mol),
                            'num_h_donors': Descriptors.NumHDonors(mol),
                            'num_h_acceptors': Descriptors.NumHAcceptors(mol),
                            'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                            'tpsa': Descriptors.TPSA(mol)
                        }
                        mol_properties.append(properties)
                
                if mol_properties:
                    # Calculate correlations with predictions
                    for prop_name in mol_properties[0].keys():
                        prop_values = [p[prop_name] for p in mol_properties]
                        if len(prop_values) == len(y_pred):
                            correlation = np.corrcoef(prop_values, y_pred)[0, 1]
                            molecular_insights['molecular_property_correlations'][prop_name] = correlation
            
            return molecular_insights
            
        except Exception as e:
            logger.error(f"Error analyzing molecular patterns: {e}")
            return {}
    
    def perform_toxicity_interpretability(self, model, X, feature_names: List[str], 
                                        y_true, y_pred) -> Dict[str, Any]:
        """Perform model interpretability analysis for toxicity classification"""
        try:
            interpretability_results = {
                'model_name': model.__class__.__name__,
                'timestamp': datetime.now(),
                'feature_importance': {},
                'partial_dependence': {},
                'shap_analysis': {},
                'lime_analysis': {},
                'toxicity_specific_insights': {}
            }
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                interpretability_results['feature_importance']['tree_based'] = {
                    'importance_scores': model.feature_importances_.tolist(),
                    'feature_names': feature_names
                }
            
            # Permutation importance
            try:
                perm_importance = permutation_importance(model, X, y_true, 
                                                       n_repeats=10, random_state=42)
                interpretability_results['feature_importance']['permutation'] = {
                    'importance_scores': perm_importance.importances_mean.tolist(),
                    'importance_std': perm_importance.importances_std.tolist(),
                    'feature_names': feature_names
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
                        X, feature_names=feature_names, class_names=['Non-toxic', 'Toxic']
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
            
            # Toxicity-specific insights
            interpretability_results['toxicity_specific_insights'] = self.extract_toxicity_insights(
                model, X, feature_names, y_true, y_pred
            )
            
            # Store results
            self.interpretability_results[model.__class__.__name__] = interpretability_results
            
            return interpretability_results
            
        except Exception as e:
            logger.error(f"Error performing toxicity interpretability: {e}")
            return {}
    
    def extract_toxicity_insights(self, model, X, feature_names: List[str], 
                                y_true, y_pred) -> Dict[str, Any]:
        """Extract toxicity-specific insights from model"""
        try:
            insights = {
                'toxic_vs_nontoxic_features': {},
                'misclassification_patterns': {},
                'confidence_patterns': {}
            }
            
            # Analyze features for toxic vs non-toxic predictions
            toxic_mask = y_pred == 1
            nontoxic_mask = y_pred == 0
            
            if hasattr(model, 'feature_importances_') and len(feature_names) == len(model.feature_importances_):
                toxic_importance = model.feature_importances_[toxic_mask] if np.any(toxic_mask) else []
                nontoxic_importance = model.feature_importances_[nontoxic_mask] if np.any(nontoxic_mask) else []
                
                insights['toxic_vs_nontoxic_features'] = {
                    'toxic_avg_importance': np.mean(toxic_importance) if len(toxic_importance) > 0 else 0,
                    'nontoxic_avg_importance': np.mean(nontoxic_importance) if len(nontoxic_importance) > 0 else 0
                }
            
            # Analyze misclassification patterns
            false_positives = (y_true == 0) & (y_pred == 1)
            false_negatives = (y_true == 1) & (y_pred == 0)
            
            insights['misclassification_patterns'] = {
                'false_positive_rate': np.mean(false_positives),
                'false_negative_rate': np.mean(false_negatives),
                'false_positive_indices': np.where(false_positives)[0].tolist(),
                'false_negative_indices': np.where(false_negatives)[0].tolist()
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error extracting toxicity insights: {e}")
            return {}
    
    def generate_toxicity_performance_plots(self, model_name: str, output_dir: str = "plots") -> List[str]:
        """Generate toxicity-specific performance visualization plots"""
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
                plt.title(f'Toxicity Classification Confusion Matrix - {model_name}')
                plt.ylabel('True Toxicity')
                plt.xlabel('Predicted Toxicity')
                
                plot_path = os.path.join(output_dir, f'{model_name}_toxicity_confusion_matrix.png')
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
                plt.title(f'Toxicity Classification ROC Curve - {model_name}')
                plt.legend()
                
                plot_path = os.path.join(output_dir, f'{model_name}_toxicity_roc_curve.png')
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
                plt.title(f'Toxicity Classification Precision-Recall Curve - {model_name}')
                plt.legend()
                
                plot_path = os.path.join(output_dir, f'{model_name}_toxicity_pr_curve.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files.append(plot_path)
            
            # Feature importance plot
            if 'feature_importance' in results.get('interpretability', {}) and results['interpretability']['feature_importance']:
                for imp_type, imp_data in results['interpretability']['feature_importance'].items():
                    if 'importance_scores' in imp_data and 'feature_names' in imp_data:
                        plt.figure(figsize=(12, 8))
                        importance_df = pd.DataFrame({
                            'feature': imp_data['feature_names'],
                            'importance': imp_data['importance_scores']
                        }).sort_values('importance', ascending=True).tail(20)
                        
                        plt.barh(range(len(importance_df)), importance_df['importance'])
                        plt.yticks(range(len(importance_df)), importance_df['feature'])
                        plt.title(f'Toxicity Feature Importance ({imp_type}) - {model_name}')
                        plt.xlabel('Importance Score')
                        
                        plot_path = os.path.join(output_dir, f'{model_name}_toxicity_{imp_type}_importance.png')
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
                    plt.title(f'Toxicity Error Type Distribution - {model_name}')
                    plt.ylabel('Count')
                    
                    plot_path = os.path.join(output_dir, f'{model_name}_toxicity_error_types.png')
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    plot_files.append(plot_path)
            
            logger.info(f"Generated {len(plot_files)} toxicity-specific plots for {model_name}")
            return plot_files
            
        except Exception as e:
            logger.error(f"Error generating toxicity performance plots for {model_name}: {e}")
            return []
    
    def generate_toxicity_evaluation_report(self, model_name: str, output_path: str = None) -> str:
        """Generate comprehensive toxicity evaluation report"""
        try:
            if model_name not in self.evaluation_results:
                return f"No evaluation results found for {model_name}"
            
            results = self.evaluation_results[model_name]
            
            report = []
            report.append("=" * 80)
            report.append(f"DRUG TOXICITY CLASSIFICATION EVALUATION REPORT - {model_name.upper()}")
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
                
                if 'toxicity_threshold_analysis' in error_analysis:
                    report.append(f"\nToxicity Threshold Analysis:")
                    threshold_analysis = error_analysis['toxicity_threshold_analysis']
                    for threshold, metrics in list(threshold_analysis.items())[:5]:  # Show first 5
                        report.append(f"  {threshold}: Accuracy={metrics['accuracy']:.3f}, Recall={metrics['recall']:.3f}")
            
            # Molecular insights
            if 'molecular_insights' in results:
                report.append("\n\nMOLECULAR INSIGHTS")
                report.append("-" * 40)
                
                mol_insights = results['molecular_insights']
                if 'toxicity_distribution' in mol_insights:
                    report.append("Toxicity Distribution:")
                    for key, count in mol_insights['toxicity_distribution'].items():
                        report.append(f"  {key}: {count}")
                
                if 'molecular_property_correlations' in mol_insights:
                    report.append(f"\nMolecular Property Correlations:")
                    for prop, corr in mol_insights['molecular_property_correlations'].items():
                        report.append(f"  {prop}: {corr:.3f}")
            
            # Interpretability summary
            if 'interpretability' in results:
                report.append("\n\nINTERPRETABILITY ANALYSIS")
                report.append("-" * 40)
                
                interpretability = results['interpretability']
                if 'feature_importance' in interpretability:
                    report.append("Feature Importance Analysis Available")
                    for imp_type in interpretability['feature_importance'].keys():
                        report.append(f"  - {imp_type} importance computed")
                
                if 'toxicity_specific_insights' in interpretability:
                    insights = interpretability['toxicity_specific_insights']
                    if 'misclassification_patterns' in insights:
                        patterns = insights['misclassification_patterns']
                        report.append(f"\nMisclassification Patterns:")
                        report.append(f"  False Positive Rate: {patterns.get('false_positive_rate', 0):.4f}")
                        report.append(f"  False Negative Rate: {patterns.get('false_negative_rate', 0):.4f}")
            
            # Toxicity-specific recommendations
            report.append("\n\nTOXICITY-SPECIFIC RECOMMENDATIONS")
            report.append("-" * 40)
            
            # Performance-based recommendations
            accuracy = metrics.get('accuracy', 0)
            f1_score = metrics.get('f1_weighted', 0)
            roc_auc = metrics.get('roc_auc', 0)
            
            if accuracy < 0.8:
                report.append("- Model accuracy is below 80%. Consider:")
                report.append("  * Collecting more diverse molecular data")
                report.append("  * Using more sophisticated molecular descriptors")
                report.append("  * Ensemble methods with different algorithms")
            
            if f1_score < 0.7:
                report.append("- F1 score indicates class imbalance issues. Consider:")
                report.append("  * Class balancing techniques (SMOTE, ADASYN)")
                report.append("  * Cost-sensitive learning")
                report.append("  * Different evaluation metrics")
            
            if roc_auc < 0.8:
                report.append("- ROC AUC indicates poor discrimination. Consider:")
                report.append("  * Feature engineering with domain knowledge")
                report.append("  * Using molecular fingerprints")
                report.append("  * Graph neural networks for molecular representation")
            
            # Toxicity-specific recommendations
            report.append("\nToxicity Classification Best Practices:")
            report.append("- Focus on high recall to avoid missing toxic compounds")
            report.append("- Use multiple molecular representations (descriptors + fingerprints)")
            report.append("- Validate on external datasets with different chemical spaces")
            report.append("- Consider ensemble methods for improved robustness")
            report.append("- Monitor for chemical domain drift over time")
            
            report_text = "\n".join(report)
            
            # Save report
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(report_text)
                logger.info(f"Toxicity evaluation report saved to {output_path}")
            
            return report_text
            
        except Exception as e:
            logger.error(f"Error generating toxicity evaluation report for {model_name}: {e}")
            return f"Error generating toxicity evaluation report for {model_name}"
    
    def compare_toxicity_models(self, model_names: List[str]) -> Dict[str, Any]:
        """Compare performance across multiple toxicity models"""
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
                metrics_to_compare = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted', 'roc_auc']
                
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
            
            logger.info("Toxicity model comparison completed")
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing toxicity models: {e}")
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
        'evaluation_metrics': ['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc'],
        'confidence_thresholds': [0.5, 0.7, 0.9],
        'plot_output_dir': 'toxicity_evaluation_plots'
    }
    
    evaluator = ToxicityModelEvaluator(config)
    
    # Example data (replace with actual molecular data)
    X = np.random.rand(100, 50)  # Molecular descriptors
    y = np.random.randint(0, 2, 100)  # Binary toxicity labels
    smiles_data = ["CC(=O)OC1=CC=CC=C1C(=O)O"] * 100  # Example SMILES
    
    # Example model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    
    # Evaluate model
    results = evaluator.evaluate_classification_model(
        model, X, y, model_name="random_forest", 
        feature_names=[f"desc_{i}" for i in range(50)],
        smiles_data=smiles_data
    )
    
    # Generate plots
    plot_files = evaluator.generate_toxicity_performance_plots("random_forest")
    
    # Generate report
    report = evaluator.generate_toxicity_evaluation_report("random_forest", "toxicity_evaluation_report.txt")
    print(report) 