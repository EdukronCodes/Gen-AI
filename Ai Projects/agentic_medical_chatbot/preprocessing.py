"""
Medical Chatbot Data Preprocessing
Handles data cleaning, feature extraction, and text preprocessing for medical conversations
"""

import os
import json
import logging
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import nltk
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalTextPreprocessor:
    """Preprocessor for medical text data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nlp = spacy.load("en_core_web_sm")
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.medical_stop_words = self.load_medical_stop_words()
        self.medical_entities = self.load_medical_entities()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_medical_stop_words(self) -> set:
        """Load medical-specific stop words"""
        medical_stop_words = {
            'patient', 'doctor', 'physician', 'nurse', 'hospital', 'clinic',
            'medical', 'health', 'care', 'treatment', 'diagnosis', 'symptom',
            'condition', 'disease', 'illness', 'medication', 'prescription',
            'appointment', 'visit', 'consultation', 'examination', 'test',
            'result', 'report', 'chart', 'record', 'history', 'family',
            'personal', 'medical', 'history', 'allergy', 'reaction'
        }
        return medical_stop_words
    
    def load_medical_entities(self) -> Dict[str, List[str]]:
        """Load medical entity categories"""
        return {
            'symptoms': [
                'pain', 'fever', 'headache', 'nausea', 'vomiting', 'diarrhea',
                'fatigue', 'weakness', 'dizziness', 'shortness of breath',
                'chest pain', 'abdominal pain', 'back pain', 'joint pain'
            ],
            'medications': [
                'aspirin', 'ibuprofen', 'acetaminophen', 'antibiotics',
                'antihistamines', 'decongestants', 'pain relievers'
            ],
            'body_parts': [
                'head', 'chest', 'abdomen', 'back', 'arms', 'legs',
                'heart', 'lungs', 'liver', 'kidneys', 'stomach'
            ],
            'conditions': [
                'hypertension', 'diabetes', 'asthma', 'arthritis',
                'depression', 'anxiety', 'cancer', 'heart disease'
            ]
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        try:
            if not text or not isinstance(text, str):
                return ""
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and extra whitespace
            text = re.sub(r'[^\w\s\.\,\!\?\-]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            
            # Remove numbers (optional - can be kept for medical contexts)
            if self.config.get('remove_numbers', False):
                text = re.sub(r'\d+', '', text)
            
            # Remove URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text if text else ""
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities from text"""
        try:
            entities = {category: [] for category in self.medical_entities.keys()}
            
            # Use spaCy for NER
            doc = self.nlp(text)
            
            for ent in doc.ents:
                if ent.label_ in ['CONDITION', 'DISEASE', 'SYMPTOM']:
                    entities['conditions'].append(ent.text.lower())
                elif ent.label_ in ['MEDICATION', 'DRUG']:
                    entities['medications'].append(ent.text.lower())
                elif ent.label_ in ['BODY_PART', 'ORGAN']:
                    entities['body_parts'].append(ent.text.lower())
            
            # Also check against predefined medical terms
            text_lower = text.lower()
            for category, terms in self.medical_entities.items():
                for term in terms:
                    if term in text_lower:
                        entities[category].append(term)
            
            # Remove duplicates
            for category in entities:
                entities[category] = list(set(entities[category]))
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting medical entities: {e}")
            return {category: [] for category in self.medical_entities.keys()}
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words"""
        try:
            tokens = word_tokenize(text)
            return [token for token in tokens if token.isalnum()]
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            return []
    
    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        """Remove stop words from tokens"""
        try:
            return [token for token in tokens if token not in self.stop_words and token not in self.medical_stop_words]
        except Exception as e:
            logger.error(f"Error removing stop words: {e}")
            return tokens
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens"""
        try:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        except Exception as e:
            logger.error(f"Error lemmatizing tokens: {e}")
            return tokens
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract comprehensive features from medical text"""
        try:
            features = {}
            
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Basic text features
            features['text_length'] = len(cleaned_text)
            features['word_count'] = len(cleaned_text.split())
            features['sentence_count'] = len(sent_tokenize(cleaned_text))
            
            # Tokenize and process
            tokens = self.tokenize_text(cleaned_text)
            tokens_no_stop = self.remove_stop_words(tokens)
            lemmatized_tokens = self.lemmatize_tokens(tokens_no_stop)
            
            # Vocabulary features
            features['unique_words'] = len(set(lemmatized_tokens))
            features['avg_word_length'] = np.mean([len(token) for token in lemmatized_tokens]) if lemmatized_tokens else 0
            
            # Medical entity features
            medical_entities = self.extract_medical_entities(cleaned_text)
            for category, entities in medical_entities.items():
                features[f'{category}_count'] = len(entities)
                features[f'{category}_present'] = 1 if entities else 0
            
            # Sentiment features (basic)
            positive_words = ['good', 'better', 'improved', 'well', 'healthy', 'fine']
            negative_words = ['bad', 'worse', 'pain', 'sick', 'ill', 'unwell']
            
            features['positive_word_count'] = sum(1 for token in lemmatized_tokens if token in positive_words)
            features['negative_word_count'] = sum(1 for token in lemmatized_tokens if token in negative_words)
            
            # Urgency indicators
            urgency_words = ['urgent', 'emergency', 'immediate', 'critical', 'severe', 'serious']
            features['urgency_score'] = sum(1 for token in lemmatized_tokens if token in urgency_words)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {}
    
    def preprocess_conversation(self, conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Preprocess a complete conversation"""
        try:
            processed_data = {
                'messages': [],
                'conversation_features': {},
                'medical_context': {}
            }
            
            # Process each message
            for message in conversation:
                processed_message = {
                    'text': message.get('text', ''),
                    'role': message.get('role', 'user'),
                    'timestamp': message.get('timestamp', datetime.now()),
                    'features': self.extract_features(message.get('text', '')),
                    'medical_entities': self.extract_medical_entities(message.get('text', ''))
                }
                processed_data['messages'].append(processed_message)
            
            # Extract conversation-level features
            processed_data['conversation_features'] = self.extract_conversation_features(processed_data['messages'])
            
            # Extract medical context
            processed_data['medical_context'] = self.extract_medical_context(processed_data['messages'])
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error preprocessing conversation: {e}")
            return {}
    
    def extract_conversation_features(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract features at conversation level"""
        try:
            features = {}
            
            # Basic conversation features
            features['message_count'] = len(messages)
            features['user_message_count'] = sum(1 for msg in messages if msg['role'] == 'user')
            features['assistant_message_count'] = sum(1 for msg in messages if msg['role'] == 'assistant')
            
            # Text features
            all_text = ' '.join([msg['text'] for msg in messages])
            features['total_text_length'] = len(all_text)
            features['total_word_count'] = len(all_text.split())
            
            # Medical entity aggregation
            all_entities = {}
            for category in self.medical_entities.keys():
                all_entities[category] = []
            
            for message in messages:
                for category, entities in message['medical_entities'].items():
                    all_entities[category].extend(entities)
            
            # Count unique entities per category
            for category, entities in all_entities.items():
                features[f'total_{category}_count'] = len(set(entities))
            
            # Conversation flow features
            if len(messages) > 1:
                features['avg_message_length'] = np.mean([len(msg['text']) for msg in messages])
                features['max_message_length'] = max([len(msg['text']) for msg in messages])
                features['min_message_length'] = min([len(msg['text']) for msg in messages])
            else:
                features['avg_message_length'] = len(messages[0]['text']) if messages else 0
                features['max_message_length'] = len(messages[0]['text']) if messages else 0
                features['min_message_length'] = len(messages[0]['text']) if messages else 0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting conversation features: {e}")
            return {}
    
    def extract_medical_context(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract medical context from conversation"""
        try:
            context = {
                'primary_symptoms': [],
                'medications_mentioned': [],
                'body_parts_affected': [],
                'conditions_discussed': [],
                'urgency_level': 'low',
                'complexity_score': 0
            }
            
            # Aggregate medical entities
            all_symptoms = []
            all_medications = []
            all_body_parts = []
            all_conditions = []
            
            for message in messages:
                entities = message['medical_entities']
                all_symptoms.extend(entities.get('symptoms', []))
                all_medications.extend(entities.get('medications', []))
                all_body_parts.extend(entities.get('body_parts', []))
                all_conditions.extend(entities.get('conditions', []))
            
            # Get most common entities
            context['primary_symptoms'] = list(set(all_symptoms))
            context['medications_mentioned'] = list(set(all_medications))
            context['body_parts_affected'] = list(set(all_body_parts))
            context['conditions_discussed'] = list(set(all_conditions))
            
            # Determine urgency level
            urgency_indicators = sum(1 for msg in messages if msg['features'].get('urgency_score', 0) > 0)
            if urgency_indicators > 2:
                context['urgency_level'] = 'high'
            elif urgency_indicators > 0:
                context['urgency_level'] = 'medium'
            
            # Calculate complexity score
            complexity_score = (
                len(context['primary_symptoms']) * 2 +
                len(context['medications_mentioned']) * 1.5 +
                len(context['conditions_discussed']) * 2.5 +
                len(context['body_parts_affected']) * 1
            )
            context['complexity_score'] = complexity_score
            
            return context
            
        except Exception as e:
            logger.error(f"Error extracting medical context: {e}")
            return {}

class MedicalDataPreprocessor:
    """Main preprocessor for medical chatbot data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.text_preprocessor = MedicalTextPreprocessor(config)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=config.get('max_features', 1000),
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.count_vectorizer = CountVectorizer(
            max_features=config.get('max_features', 500),
            ngram_range=(1, 1)
        )
        
    def preprocess_training_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Preprocess training data for model training"""
        try:
            processed_data = {
                'texts': [],
                'features': [],
                'labels': [],
                'conversations': []
            }
            
            for item in data:
                # Process conversation
                conversation = item.get('conversation', [])
                processed_conversation = self.text_preprocessor.preprocess_conversation(conversation)
                
                # Extract text for vectorization
                all_text = ' '.join([msg.get('text', '') for msg in conversation])
                processed_data['texts'].append(all_text)
                
                # Extract features
                features = processed_conversation.get('conversation_features', {})
                processed_data['features'].append(features)
                
                # Extract labels
                label = item.get('label', 'general')
                processed_data['labels'].append(label)
                
                # Store processed conversation
                processed_data['conversations'].append(processed_conversation)
            
            # Vectorize text
            if processed_data['texts']:
                tfidf_features = self.tfidf_vectorizer.fit_transform(processed_data['texts'])
                count_features = self.count_vectorizer.fit_transform(processed_data['texts'])
                
                processed_data['tfidf_features'] = tfidf_features.toarray()
                processed_data['count_features'] = count_features.toarray()
            
            # Encode labels
            label_encoder = LabelEncoder()
            processed_data['encoded_labels'] = label_encoder.fit_transform(processed_data['labels'])
            processed_data['label_encoder'] = label_encoder
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error preprocessing training data: {e}")
            return {}
    
    def preprocess_single_query(self, query: str) -> Dict[str, Any]:
        """Preprocess a single user query"""
        try:
            processed_data = {}
            
            # Extract features
            features = self.text_preprocessor.extract_features(query)
            processed_data['features'] = features
            
            # Extract medical entities
            medical_entities = self.text_preprocessor.extract_medical_entities(query)
            processed_data['medical_entities'] = medical_entities
            
            # Vectorize text
            if hasattr(self.tfidf_vectorizer, 'vocabulary_'):
                tfidf_features = self.tfidf_vectorizer.transform([query])
                count_features = self.count_vectorizer.transform([query])
                
                processed_data['tfidf_features'] = tfidf_features.toarray()
                processed_data['count_features'] = count_features.toarray()
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error preprocessing single query: {e}")
            return {}
    
    def save_preprocessor(self, filepath: str):
        """Save preprocessor state"""
        try:
            preprocessor_state = {
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'count_vectorizer': self.count_vectorizer,
                'config': self.config
            }
            
            with open(filepath, 'wb') as f:
                import pickle
                pickle.dump(preprocessor_state, f)
            
            logger.info(f"Preprocessor saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving preprocessor: {e}")
    
    def load_preprocessor(self, filepath: str):
        """Load preprocessor state"""
        try:
            with open(filepath, 'rb') as f:
                import pickle
                preprocessor_state = pickle.load(f)
            
            self.tfidf_vectorizer = preprocessor_state['tfidf_vectorizer']
            self.count_vectorizer = preprocessor_state['count_vectorizer']
            self.config = preprocessor_state['config']
            
            logger.info(f"Preprocessor loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading preprocessor: {e}")

# Example usage
if __name__ == "__main__":
    config = {
        'max_features': 1000,
        'remove_numbers': False,
        'medical_stop_words': True
    }
    
    preprocessor = MedicalDataPreprocessor(config)
    
    # Example conversation
    conversation = [
        {'role': 'user', 'text': 'I have been experiencing severe chest pain for the past 2 days.'},
        {'role': 'assistant', 'text': 'I understand you are experiencing chest pain. Can you tell me more about the symptoms?'},
        {'role': 'user', 'text': 'The pain is sharp and radiates to my left arm. I also feel shortness of breath.'}
    ]
    
    # Preprocess conversation
    processed = preprocessor.text_preprocessor.preprocess_conversation(conversation)
    print("Processed conversation features:", processed['conversation_features'])
    print("Medical context:", processed['medical_context']) 