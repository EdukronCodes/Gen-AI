"""
Patient Stratification Healthcare Data Preprocessing
Handles patient data preprocessing, feature extraction, and clinical data cleaning
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Medical/Clinical imports
import re
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatientDataPreprocessor:
    """Preprocessor for patient stratification data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.pca = None
        self.imputer = SimpleImputer(strategy='mean')
        self.label_encoders = {}
        self.feature_names = []
        self.clinical_variables = {}
        
    def load_clinical_variables(self) -> Dict[str, List[str]]:
        """Load clinical variable categories"""
        return {
            'demographics': [
                'age', 'gender', 'race', 'ethnicity', 'marital_status',
                'education_level', 'income_level', 'insurance_type'
            ],
            'vital_signs': [
                'blood_pressure_systolic', 'blood_pressure_diastolic',
                'heart_rate', 'respiratory_rate', 'temperature',
                'oxygen_saturation', 'bmi', 'weight', 'height'
            ],
            'lab_results': [
                'glucose', 'creatinine', 'bun', 'sodium', 'potassium',
                'chloride', 'bicarbonate', 'hemoglobin', 'white_blood_cells',
                'platelets', 'cholesterol_total', 'hdl', 'ldl', 'triglycerides'
            ],
            'medications': [
                'medication_count', 'antihypertensive', 'antidiabetic',
                'statin', 'aspirin', 'beta_blocker', 'ace_inhibitor'
            ],
            'comorbidities': [
                'diabetes', 'hypertension', 'heart_disease', 'kidney_disease',
                'liver_disease', 'cancer', 'copd', 'asthma', 'depression',
                'anxiety', 'obesity', 'smoking_status'
            ],
            'clinical_scores': [
                'charlson_comorbidity_index', 'elixhauser_comorbidity_index',
                'braden_scale', 'morse_fall_scale', 'braden_scale_score'
            ]
        }
    
    def clean_patient_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate patient data"""
        try:
            cleaned_data = data.copy()
            
            # Remove duplicate records
            cleaned_data = cleaned_data.drop_duplicates()
            
            # Handle missing values
            cleaned_data = self.handle_missing_values(cleaned_data)
            
            # Validate data ranges
            cleaned_data = self.validate_data_ranges(cleaned_data)
            
            # Remove outliers
            cleaned_data = self.remove_outliers(cleaned_data)
            
            # Standardize formats
            cleaned_data = self.standardize_formats(cleaned_data)
            
            logger.info(f"Cleaned data shape: {cleaned_data.shape}")
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Error cleaning patient data: {e}")
            return data
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in patient data"""
        try:
            # Check missing values
            missing_counts = data.isnull().sum()
            logger.info(f"Missing values per column: {missing_counts[missing_counts > 0]}")
            
            # Different strategies for different types of variables
            clinical_vars = self.load_clinical_variables()
            
            # Demographics - mode imputation
            for col in clinical_vars['demographics']:
                if col in data.columns and data[col].isnull().sum() > 0:
                    mode_value = data[col].mode()[0] if len(data[col].mode()) > 0 else 'Unknown'
                    data[col].fillna(mode_value, inplace=True)
            
            # Vital signs - median imputation
            for col in clinical_vars['vital_signs']:
                if col in data.columns and data[col].isnull().sum() > 0:
                    median_value = data[col].median()
                    data[col].fillna(median_value, inplace=True)
            
            # Lab results - KNN imputation for correlated variables
            lab_cols = [col for col in clinical_vars['lab_results'] if col in data.columns]
            if lab_cols and data[lab_cols].isnull().sum().sum() > 0:
                knn_imputer = KNNImputer(n_neighbors=5)
                data[lab_cols] = knn_imputer.fit_transform(data[lab_cols])
            
            # Medications and comorbidities - fill with 0 (no medication/disease)
            for col in clinical_vars['medications'] + clinical_vars['comorbidities']:
                if col in data.columns and data[col].isnull().sum() > 0:
                    data[col].fillna(0, inplace=True)
            
            # Clinical scores - mean imputation
            for col in clinical_vars['clinical_scores']:
                if col in data.columns and data[col].isnull().sum() > 0:
                    mean_value = data[col].mean()
                    data[col].fillna(mean_value, inplace=True)
            
            return data
            
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            return data
    
    def validate_data_ranges(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate data ranges for clinical variables"""
        try:
            clinical_vars = self.load_clinical_variables()
            
            # Age validation
            if 'age' in data.columns:
                data = data[(data['age'] >= 0) & (data['age'] <= 120)]
            
            # Vital signs validation
            vital_ranges = {
                'blood_pressure_systolic': (70, 250),
                'blood_pressure_diastolic': (40, 150),
                'heart_rate': (30, 200),
                'respiratory_rate': (8, 50),
                'temperature': (35, 42),
                'oxygen_saturation': (70, 100),
                'bmi': (10, 80),
                'weight': (20, 500),  # kg
                'height': (100, 250)   # cm
            }
            
            for col, (min_val, max_val) in vital_ranges.items():
                if col in data.columns:
                    data = data[(data[col] >= min_val) & (data[col] <= max_val)]
            
            # Lab results validation
            lab_ranges = {
                'glucose': (20, 1000),      # mg/dL
                'creatinine': (0.1, 20),    # mg/dL
                'bun': (1, 200),            # mg/dL
                'sodium': (110, 180),       # mEq/L
                'potassium': (1, 10),       # mEq/L
                'chloride': (70, 130),      # mEq/L
                'bicarbonate': (10, 40),    # mEq/L
                'hemoglobin': (3, 25),      # g/dL
                'white_blood_cells': (0.1, 100),  # K/μL
                'platelets': (10, 1000),    # K/μL
                'cholesterol_total': (50, 800),   # mg/dL
                'hdl': (10, 200),           # mg/dL
                'ldl': (20, 400),           # mg/dL
                'triglycerides': (20, 2000) # mg/dL
            }
            
            for col, (min_val, max_val) in lab_ranges.items():
                if col in data.columns:
                    data = data[(data[col] >= min_val) & (data[col] <= max_val)]
            
            return data
            
        except Exception as e:
            logger.error(f"Error validating data ranges: {e}")
            return data
    
    def remove_outliers(self, data: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Remove outliers from numerical variables"""
        try:
            clinical_vars = self.load_clinical_variables()
            numerical_cols = clinical_vars['vital_signs'] + clinical_vars['lab_results'] + clinical_vars['clinical_scores']
            
            for col in numerical_cols:
                if col in data.columns:
                    if method == 'iqr':
                        Q1 = data[col].quantile(0.25)
                        Q3 = data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
                    elif method == 'zscore':
                        z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                        data = data[z_scores < 3]
            
            return data
            
        except Exception as e:
            logger.error(f"Error removing outliers: {e}")
            return data
    
    def standardize_formats(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize data formats"""
        try:
            # Standardize date formats
            date_columns = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
            for col in date_columns:
                if col in data.columns:
                    data[col] = pd.to_datetime(data[col], errors='coerce')
            
            # Standardize categorical variables
            categorical_cols = ['gender', 'race', 'ethnicity', 'marital_status', 'insurance_type']
            for col in categorical_cols:
                if col in data.columns:
                    data[col] = data[col].astype('category')
            
            # Convert boolean columns
            boolean_cols = self.load_clinical_variables()['comorbidities'] + self.load_clinical_variables()['medications']
            for col in boolean_cols:
                if col in data.columns:
                    data[col] = data[col].astype(bool)
            
            return data
            
        except Exception as e:
            logger.error(f"Error standardizing formats: {e}")
            return data
    
    def extract_clinical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract clinical features from patient data"""
        try:
            features = data.copy()
            clinical_vars = self.load_clinical_variables()
            
            # Age-related features
            if 'age' in features.columns:
                features['age_group'] = pd.cut(features['age'], 
                                             bins=[0, 18, 35, 50, 65, 80, 120], 
                                             labels=['child', 'young_adult', 'adult', 'middle_aged', 'elderly', 'very_elderly'])
                features['age_squared'] = features['age'] ** 2
            
            # BMI categories
            if 'bmi' in features.columns:
                features['bmi_category'] = pd.cut(features['bmi'],
                                                bins=[0, 18.5, 25, 30, 35, 100],
                                                labels=['underweight', 'normal', 'overweight', 'obese', 'severely_obese'])
            
            # Blood pressure categories
            if 'blood_pressure_systolic' in features.columns and 'blood_pressure_diastolic' in features.columns:
                features['bp_category'] = self.categorize_blood_pressure(
                    features['blood_pressure_systolic'], features['blood_pressure_diastolic']
                )
            
            # Lab result ratios and indices
            if 'hdl' in features.columns and 'ldl' in features.columns:
                features['hdl_ldl_ratio'] = features['hdl'] / features['ldl']
            
            if 'cholesterol_total' in features.columns and 'hdl' in features.columns:
                features['total_hdl_ratio'] = features['cholesterol_total'] / features['hdl']
            
            # Comorbidity count
            comorbidity_cols = [col for col in clinical_vars['comorbidities'] if col in features.columns]
            if comorbidity_cols:
                features['comorbidity_count'] = features[comorbidity_cols].sum(axis=1)
            
            # Medication count
            medication_cols = [col for col in clinical_vars['medications'] if col in features.columns]
            if medication_cols:
                features['medication_count'] = features[medication_cols].sum(axis=1)
            
            # Risk scores
            features = self.calculate_risk_scores(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting clinical features: {e}")
            return data
    
    def categorize_blood_pressure(self, systolic: pd.Series, diastolic: pd.Series) -> pd.Series:
        """Categorize blood pressure readings"""
        try:
            categories = []
            for sys, dia in zip(systolic, diastolic):
                if sys < 120 and dia < 80:
                    categories.append('normal')
                elif (120 <= sys < 130) and dia < 80:
                    categories.append('elevated')
                elif (130 <= sys < 140) or (80 <= dia < 90):
                    categories.append('stage1_hypertension')
                elif sys >= 140 or dia >= 90:
                    categories.append('stage2_hypertension')
                else:
                    categories.append('unknown')
            
            return pd.Series(categories)
            
        except Exception as e:
            logger.error(f"Error categorizing blood pressure: {e}")
            return pd.Series(['unknown'] * len(systolic))
    
    def calculate_risk_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate clinical risk scores"""
        try:
            # Framingham Risk Score (simplified)
            if all(col in data.columns for col in ['age', 'gender', 'cholesterol_total', 'hdl', 'blood_pressure_systolic', 'smoking_status']):
                data['framingham_risk_score'] = self.calculate_framingham_score(data)
            
            # CHA2DS2-VASc Score (for atrial fibrillation risk)
            if all(col in data.columns for col in ['age', 'gender', 'heart_disease', 'diabetes', 'hypertension']):
                data['cha2ds2_vasc_score'] = self.calculate_cha2ds2_vasc_score(data)
            
            # HAS-BLED Score (for bleeding risk)
            if all(col in data.columns for col in ['hypertension', 'kidney_disease', 'liver_disease']):
                data['has_bled_score'] = self.calculate_has_bled_score(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error calculating risk scores: {e}")
            return data
    
    def calculate_framingham_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate simplified Framingham Risk Score"""
        try:
            scores = []
            for _, row in data.iterrows():
                score = 0
                
                # Age scoring
                if row['gender'] == 'male':
                    if 20 <= row['age'] <= 34:
                        score += -9
                    elif 35 <= row['age'] <= 39:
                        score += -4
                    elif 40 <= row['age'] <= 44:
                        score += 0
                    elif 45 <= row['age'] <= 49:
                        score += 3
                    elif 50 <= row['age'] <= 54:
                        score += 6
                    elif 55 <= row['age'] <= 59:
                        score += 8
                    elif 60 <= row['age'] <= 64:
                        score += 10
                    elif 65 <= row['age'] <= 69:
                        score += 11
                    elif 70 <= row['age'] <= 74:
                        score += 12
                    else:
                        score += 13
                
                # Cholesterol scoring (simplified)
                if row['cholesterol_total'] >= 240:
                    score += 1
                
                # Blood pressure scoring
                if row['blood_pressure_systolic'] >= 140:
                    score += 1
                
                # Smoking scoring
                if row['smoking_status']:
                    score += 2
                
                scores.append(score)
            
            return pd.Series(scores)
            
        except Exception as e:
            logger.error(f"Error calculating Framingham score: {e}")
            return pd.Series([0] * len(data))
    
    def calculate_cha2ds2_vasc_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate CHA2DS2-VASc Score"""
        try:
            scores = []
            for _, row in data.iterrows():
                score = 0
                
                # Age scoring
                if 65 <= row['age'] <= 74:
                    score += 1
                elif row['age'] >= 75:
                    score += 2
                
                # Gender scoring
                if row['gender'] == 'female':
                    score += 1
                
                # Comorbidity scoring
                if row['heart_disease']:
                    score += 1
                if row['diabetes']:
                    score += 1
                if row['hypertension']:
                    score += 1
                
                scores.append(score)
            
            return pd.Series(scores)
            
        except Exception as e:
            logger.error(f"Error calculating CHA2DS2-VASc score: {e}")
            return pd.Series([0] * len(data))
    
    def calculate_has_bled_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate HAS-BLED Score"""
        try:
            scores = []
            for _, row in data.iterrows():
                score = 0
                
                # Comorbidity scoring
                if row['hypertension']:
                    score += 1
                if row['kidney_disease']:
                    score += 1
                if row['liver_disease']:
                    score += 1
                
                scores.append(score)
            
            return pd.Series(scores)
            
        except Exception as e:
            logger.error(f"Error calculating HAS-BLED score: {e}")
            return pd.Series([0] * len(data))
    
    def encode_categorical_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        try:
            encoded_data = data.copy()
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            
            for col in categorical_cols:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    encoded_data[col] = self.label_encoders[col].fit_transform(encoded_data[col].astype(str))
                else:
                    # Handle new categories
                    unique_values = set(encoded_data[col].unique())
                    known_values = set(self.label_encoders[col].classes_)
                    new_values = unique_values - known_values
                    
                    if new_values:
                        # Add new categories
                        all_values = list(known_values) + list(new_values)
                        self.label_encoders[col] = LabelEncoder()
                        self.label_encoders[col].fit(all_values)
                    
                    encoded_data[col] = self.label_encoders[col].transform(encoded_data[col].astype(str))
            
            return encoded_data
            
        except Exception as e:
            logger.error(f"Error encoding categorical variables: {e}")
            return data
    
    def scale_features(self, data: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Scale numerical features"""
        try:
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            scaled_data = data.copy()
            
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
            elif method == 'robust':
                self.scaler = RobustScaler()
            
            scaled_data[numerical_cols] = self.scaler.fit_transform(data[numerical_cols])
            
            return scaled_data
            
        except Exception as e:
            logger.error(f"Error scaling features: {e}")
            return data
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'mutual_info', 
                       k: int = 50) -> pd.DataFrame:
        """Select most important features"""
        try:
            if method == 'mutual_info':
                selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
            elif method == 'f_classif':
                selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
            elif method == 'rfe':
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)
                selector = RFE(estimator=estimator, n_features_to_select=min(k, X.shape[1]))
            else:
                logger.warning(f"Unknown feature selection method: {method}")
                return X
            
            selected_features = selector.fit_transform(X, y)
            selected_columns = X.columns[selector.get_support()].tolist()
            
            selected_df = pd.DataFrame(
                selected_features,
                columns=selected_columns,
                index=X.index
            )
            
            self.feature_selector = selector
            self.feature_names = selected_columns
            
            logger.info(f"Selected {len(selected_columns)} features using {method}")
            
            return selected_df
            
        except Exception as e:
            logger.error(f"Error selecting features: {e}")
            return X
    
    def preprocess_patient_data(self, data: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """Complete preprocessing pipeline for patient data"""
        try:
            processed_data = {
                'raw_data': data,
                'cleaned_data': None,
                'feature_data': None,
                'encoded_data': None,
                'scaled_data': None,
                'selected_features': None,
                'target': None,
                'feature_names': []
            }
            
            # Clean data
            logger.info("Cleaning patient data...")
            cleaned_data = self.clean_patient_data(data)
            processed_data['cleaned_data'] = cleaned_data
            
            # Extract features
            logger.info("Extracting clinical features...")
            feature_data = self.extract_clinical_features(cleaned_data)
            processed_data['feature_data'] = feature_data
            
            # Separate target variable
            if target_column and target_column in feature_data.columns:
                processed_data['target'] = feature_data[target_column]
                feature_data = feature_data.drop(columns=[target_column])
            
            # Encode categorical variables
            logger.info("Encoding categorical variables...")
            encoded_data = self.encode_categorical_variables(feature_data)
            processed_data['encoded_data'] = encoded_data
            
            # Scale features
            logger.info("Scaling features...")
            scaled_data = self.scale_features(encoded_data)
            processed_data['scaled_data'] = scaled_data
            
            # Feature selection
            if processed_data['target'] is not None:
                logger.info("Selecting features...")
                selected_features = self.select_features(scaled_data, processed_data['target'])
                processed_data['selected_features'] = selected_features
                processed_data['feature_names'] = self.feature_names
            
            logger.info(f"Preprocessing completed. Final shape: {processed_data['selected_features'].shape if processed_data['selected_features'] is not None else processed_data['scaled_data'].shape}")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {e}")
            return {}
    
    def save_preprocessor(self, filepath: str):
        """Save preprocessor state"""
        try:
            preprocessor_state = {
                'scaler': self.scaler,
                'imputer': self.imputer,
                'pca': self.pca,
                'feature_selector': self.feature_selector,
                'label_encoders': self.label_encoders,
                'config': self.config,
                'feature_names': self.feature_names
            }
            
            joblib.dump(preprocessor_state, filepath)
            logger.info(f"Preprocessor saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving preprocessor: {e}")
    
    def load_preprocessor(self, filepath: str):
        """Load preprocessor state"""
        try:
            preprocessor_state = joblib.load(filepath)
            
            self.scaler = preprocessor_state['scaler']
            self.imputer = preprocessor_state['imputer']
            self.pca = preprocessor_state['pca']
            self.feature_selector = preprocessor_state['feature_selector']
            self.label_encoders = preprocessor_state['label_encoders']
            self.config = preprocessor_state['config']
            self.feature_names = preprocessor_state['feature_names']
            
            logger.info(f"Preprocessor loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading preprocessor: {e}")

# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'feature_selection_method': 'mutual_info',
        'n_features': 50,
        'scaling_method': 'standard',
        'outlier_method': 'iqr'
    }
    
    preprocessor = PatientDataPreprocessor(config)
    
    # Example data (replace with actual patient data)
    sample_data = pd.DataFrame({
        'age': [65, 45, 72, 38, 55],
        'gender': ['male', 'female', 'male', 'female', 'male'],
        'blood_pressure_systolic': [140, 120, 160, 110, 135],
        'blood_pressure_diastolic': [90, 80, 95, 70, 85],
        'glucose': [120, 95, 150, 85, 110],
        'cholesterol_total': [220, 180, 250, 160, 200],
        'hdl': [45, 55, 40, 60, 50],
        'diabetes': [1, 0, 1, 0, 0],
        'hypertension': [1, 0, 1, 0, 1],
        'target': [1, 0, 1, 0, 0]
    })
    
    # Preprocess data
    processed = preprocessor.preprocess_patient_data(sample_data, target_column='target')
    
    print(f"Processed data shape: {processed['selected_features'].shape}")
    print(f"Feature names: {processed['feature_names']}")
    
    # Preprocess single patient
    single_patient = pd.DataFrame({
        'age': [60],
        'gender': ['male'],
        'blood_pressure_systolic': [145],
        'blood_pressure_diastolic': [88],
        'glucose': [125],
        'cholesterol_total': [230],
        'hdl': [42],
        'diabetes': [1],
        'hypertension': [1]
    })
    
    # Apply preprocessing pipeline to single patient
    single_processed = preprocessor.preprocess_patient_data(single_patient)
    print(f"Single patient processed shape: {single_processed['scaled_data'].shape}") 