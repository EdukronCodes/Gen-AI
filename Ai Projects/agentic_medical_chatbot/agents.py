"""
Additional Medical Agents for Treatment Recommendations and Emergency Assessment
Extends the base medical agent system with specialized agents
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import spacy
from transformers import pipeline
import numpy as np

from rag_medical_chatbot import MedicalAgent, MedicalQuery, MedicalResponse

class TreatmentRecommendationAgent(MedicalAgent):
    """Agent for providing treatment recommendations based on symptoms and conditions"""
    
    def __init__(self, rag_system):
        super().__init__("treatment_recommendation", rag_system)
        self.treatment_classifier = pipeline(
            "text-classification",
            model="medical-bert-treatment-classifier"
        )
        self.condition_analyzer = self.load_condition_analyzer()
    
    async def process_query(self, query: MedicalQuery) -> MedicalResponse:
        """Provide treatment recommendations based on symptoms and conditions"""
        
        # Extract conditions and symptoms from query
        conditions = self.extract_medical_conditions(query.query_text)
        symptoms = self.extract_symptoms(query.query_text)
        
        # Retrieve treatment information
        treatment_info = []
        for condition in conditions:
            info = self.rag_system.retrieve_relevant_info(
                f"treatment {condition} evidence-based", top_k=3
            )
            treatment_info.extend(info)
        
        # Analyze patient context
        patient_context = self.analyze_patient_context(query.context)
        
        # Generate treatment recommendations
        recommendations = self.generate_treatment_recommendations(
            conditions, symptoms, treatment_info, patient_context
        )
        
        # Calculate confidence score
        confidence = self.calculate_treatment_confidence(
            conditions, treatment_info, patient_context
        )
        
        return MedicalResponse(
            response_text=self.generate_treatment_response(
                conditions, symptoms, recommendations, treatment_info
            ),
            confidence_score=confidence,
            sources=[info["source"] for info in treatment_info],
            recommendations=recommendations,
            escalation_needed=confidence < 0.7,
            follow_up_questions=self.generate_treatment_questions(conditions, symptoms)
        )
    
    def extract_medical_conditions(self, text: str) -> List[str]:
        """Extract medical conditions from text"""
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        
        conditions = []
        for ent in doc.ents:
            if ent.label_ in ["DISEASE", "CONDITION", "DIAGNOSIS"]:
                conditions.append(ent.text)
        
        return conditions
    
    def extract_symptoms(self, text: str) -> List[str]:
        """Extract symptoms from text"""
        symptom_keywords = [
            "pain", "fever", "cough", "headache", "nausea", "vomiting",
            "fatigue", "dizziness", "shortness of breath", "chest pain"
        ]
        
        symptoms = []
        text_lower = text.lower()
        for symptom in symptom_keywords:
            if symptom in text_lower:
                symptoms.append(symptom)
        
        return symptoms
    
    def load_condition_analyzer(self):
        """Load condition analysis model"""
        # This would be a pre-trained model for condition analysis
        return lambda text: self.simple_condition_analysis(text)
    
    def simple_condition_analysis(self, text: str) -> Dict[str, Any]:
        """Simple condition analysis"""
        # In a real system, this would use a sophisticated medical NLP model
        return {
            "severity": "moderate",
            "chronic": False,
            "urgent": False
        }
    
    def analyze_patient_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patient context for treatment recommendations"""
        analysis = {
            "age_group": context.get("age_group", "adult"),
            "comorbidities": context.get("comorbidities", []),
            "medications": context.get("current_medications", []),
            "allergies": context.get("allergies", []),
            "pregnancy_status": context.get("pregnancy_status", "unknown")
        }
        
        return analysis
    
    def generate_treatment_recommendations(self, conditions: List[str], 
                                         symptoms: List[str], 
                                         treatment_info: List[Dict], 
                                         patient_context: Dict[str, Any]) -> List[str]:
        """Generate evidence-based treatment recommendations"""
        recommendations = []
        
        # Add general recommendations
        recommendations.append("Consult with a healthcare provider for personalized treatment")
        
        # Add condition-specific recommendations
        for condition in conditions:
            if "diabetes" in condition.lower():
                recommendations.extend([
                    "Monitor blood glucose levels regularly",
                    "Follow a balanced diet plan",
                    "Engage in regular physical activity"
                ])
            elif "hypertension" in condition.lower():
                recommendations.extend([
                    "Monitor blood pressure regularly",
                    "Reduce sodium intake",
                    "Maintain healthy weight"
                ])
            elif "asthma" in condition.lower():
                recommendations.extend([
                    "Use prescribed inhalers as directed",
                    "Avoid known triggers",
                    "Keep rescue inhaler accessible"
                ])
        
        # Add symptom-specific recommendations
        for symptom in symptoms:
            if "pain" in symptom:
                recommendations.append("Consider over-the-counter pain relievers as directed")
            if "fever" in symptom:
                recommendations.append("Rest and stay hydrated")
            if "cough" in symptom:
                recommendations.append("Use honey or over-the-counter cough suppressants")
        
        # Consider patient context
        if patient_context.get("pregnancy_status") == "pregnant":
            recommendations.append("Consult obstetrician before taking any medications")
        
        return list(set(recommendations))  # Remove duplicates
    
    def calculate_treatment_confidence(self, conditions: List[str], 
                                     treatment_info: List[Dict], 
                                     patient_context: Dict[str, Any]) -> float:
        """Calculate confidence score for treatment recommendations"""
        base_confidence = 0.7
        
        # Increase confidence based on available information
        if conditions:
            base_confidence += 0.1
        if treatment_info:
            base_confidence += 0.1
        if patient_context.get("comorbidities"):
            base_confidence += 0.05
        
        # Decrease confidence for complex cases
        if len(patient_context.get("comorbidities", [])) > 2:
            base_confidence -= 0.1
        if patient_context.get("pregnancy_status") == "pregnant":
            base_confidence -= 0.05
        
        return min(max(base_confidence, 0.0), 1.0)
    
    def generate_treatment_response(self, conditions: List[str], 
                                  symptoms: List[str], 
                                  recommendations: List[str], 
                                  treatment_info: List[Dict]) -> str:
        """Generate comprehensive treatment response"""
        response_template = """
        Treatment Recommendations
        
        Conditions Identified: {conditions}
        Symptoms: {symptoms}
        
        Evidence-Based Recommendations:
        {recommendations}
        
        Additional Information:
        {additional_info}
        
        Important Notes:
        - These recommendations are for informational purposes only
        - Always consult with a healthcare provider for personalized treatment
        - Follow up with your doctor for ongoing care
        """
        
        additional_info = "\n".join([
            f"- {item['content'][:200]}..." for item in treatment_info[:2]
        ])
        
        return response_template.format(
            conditions=", ".join(conditions) if conditions else "None identified",
            symptoms=", ".join(symptoms) if symptoms else "None specified",
            recommendations="\n".join([f"• {rec}" for rec in recommendations]),
            additional_info=additional_info
        )
    
    def generate_treatment_questions(self, conditions: List[str], symptoms: List[str]) -> List[str]:
        """Generate follow-up questions for treatment planning"""
        questions = [
            "Have you tried any treatments before?",
            "What medications are you currently taking?",
            "Do you have any allergies to medications?",
            "How long have you been experiencing these symptoms?"
        ]
        
        # Add condition-specific questions
        for condition in conditions:
            if "diabetes" in condition.lower():
                questions.append("What is your current blood glucose level?")
            elif "hypertension" in condition.lower():
                questions.append("What is your current blood pressure reading?")
        
        return questions

class EmergencyAssessmentAgent(MedicalAgent):
    """Agent for emergency assessment and triage"""
    
    def __init__(self, rag_system):
        super().__init__("emergency_assessment", rag_system)
        self.emergency_classifier = pipeline(
            "text-classification",
            model="emergency-triage-classifier"
        )
        self.vital_signs_analyzer = self.load_vital_signs_analyzer()
    
    async def process_query(self, query: MedicalQuery) -> MedicalResponse:
        """Assess emergency situation and provide immediate guidance"""
        
        # Assess emergency level
        emergency_level = self.assess_emergency_level(query.query_text)
        
        # Extract vital signs if mentioned
        vital_signs = self.extract_vital_signs(query.query_text)
        
        # Retrieve emergency protocols
        emergency_info = self.rag_system.retrieve_relevant_info(
            f"emergency {query.query_text} protocol", top_k=3
        )
        
        # Generate emergency response
        response_text = self.generate_emergency_response(
            query.query_text, emergency_level, vital_signs, emergency_info
        )
        
        # Determine if immediate escalation is needed
        escalation_needed = emergency_level >= 4 or query.urgency_level >= 4
        
        return MedicalResponse(
            response_text=response_text,
            confidence_score=0.95 if emergency_level >= 4 else 0.85,
            sources=[info["source"] for info in emergency_info],
            recommendations=self.generate_emergency_recommendations(emergency_level, vital_signs),
            escalation_needed=escalation_needed,
            follow_up_questions=self.generate_emergency_questions(query.query_text)
        )
    
    def assess_emergency_level(self, text: str) -> int:
        """Assess emergency level on a 1-5 scale"""
        text_lower = text.lower()
        
        # Critical emergency keywords
        critical_keywords = [
            "chest pain", "heart attack", "stroke", "unconscious",
            "not breathing", "severe bleeding", "seizure"
        ]
        
        # High emergency keywords
        high_keywords = [
            "difficulty breathing", "severe pain", "high fever",
            "head injury", "broken bone", "allergic reaction"
        ]
        
        # Moderate emergency keywords
        moderate_keywords = [
            "moderate pain", "persistent vomiting", "dehydration",
            "dizziness", "fainting"
        ]
        
        if any(keyword in text_lower for keyword in critical_keywords):
            return 5
        elif any(keyword in text_lower for keyword in high_keywords):
            return 4
        elif any(keyword in text_lower for keyword in moderate_keywords):
            return 3
        else:
            return 2
    
    def extract_vital_signs(self, text: str) -> Dict[str, Any]:
        """Extract vital signs from text"""
        vital_signs = {}
        
        # Extract temperature
        import re
        temp_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:degrees?|°|F|C)', text, re.IGNORECASE)
        if temp_match:
            vital_signs["temperature"] = float(temp_match.group(1))
        
        # Extract blood pressure
        bp_match = re.search(r'(\d+)/(\d+)\s*(?:blood pressure|BP)', text, re.IGNORECASE)
        if bp_match:
            vital_signs["blood_pressure"] = {
                "systolic": int(bp_match.group(1)),
                "diastolic": int(bp_match.group(2))
            }
        
        # Extract heart rate
        hr_match = re.search(r'(\d+)\s*(?:bpm|heart rate|pulse)', text, re.IGNORECASE)
        if hr_match:
            vital_signs["heart_rate"] = int(hr_match.group(1))
        
        return vital_signs
    
    def load_vital_signs_analyzer(self):
        """Load vital signs analysis model"""
        return lambda vitals: self.analyze_vital_signs(vitals)
    
    def analyze_vital_signs(self, vital_signs: Dict[str, Any]) -> Dict[str, str]:
        """Analyze vital signs for abnormalities"""
        analysis = {}
        
        if "temperature" in vital_signs:
            temp = vital_signs["temperature"]
            if temp > 103:
                analysis["temperature"] = "critical_high"
            elif temp > 100.4:
                analysis["temperature"] = "elevated"
            else:
                analysis["temperature"] = "normal"
        
        if "blood_pressure" in vital_signs:
            bp = vital_signs["blood_pressure"]
            if bp["systolic"] > 180 or bp["diastolic"] > 110:
                analysis["blood_pressure"] = "critical_high"
            elif bp["systolic"] > 140 or bp["diastolic"] > 90:
                analysis["blood_pressure"] = "elevated"
            else:
                analysis["blood_pressure"] = "normal"
        
        if "heart_rate" in vital_signs:
            hr = vital_signs["heart_rate"]
            if hr > 100:
                analysis["heart_rate"] = "tachycardia"
            elif hr < 60:
                analysis["heart_rate"] = "bradycardia"
            else:
                analysis["heart_rate"] = "normal"
        
        return analysis
    
    def generate_emergency_response(self, query: str, emergency_level: int, 
                                  vital_signs: Dict[str, Any], 
                                  emergency_info: List[Dict]) -> str:
        """Generate emergency response with appropriate urgency"""
        
        if emergency_level == 5:
            response_template = """
            🚨 CRITICAL EMERGENCY 🚨
            
            Your symptoms indicate a critical emergency requiring IMMEDIATE medical attention.
            
            IMMEDIATE ACTIONS:
            • Call 911 or emergency services immediately
            • Do not drive yourself to the hospital
            • Stay calm and follow emergency operator instructions
            
            Symptoms: {symptoms}
            
            {vital_signs_info}
            
            {emergency_protocol}
            """
        elif emergency_level == 4:
            response_template = """
            ⚠️ HIGH URGENCY EMERGENCY ⚠️
            
            Your symptoms require urgent medical attention.
            
            RECOMMENDED ACTIONS:
            • Seek immediate medical care
            • Go to nearest emergency room or urgent care
            • Do not delay treatment
            
            Symptoms: {symptoms}
            
            {vital_signs_info}
            
            {emergency_protocol}
            """
        else:
            response_template = """
            Medical Assessment
            
            Your symptoms require medical attention but are not immediately life-threatening.
            
            RECOMMENDED ACTIONS:
            • Contact your healthcare provider
            • Monitor symptoms closely
            • Seek care if symptoms worsen
            
            Symptoms: {symptoms}
            
            {vital_signs_info}
            
            {emergency_protocol}
            """
        
        vital_signs_info = ""
        if vital_signs:
            vital_signs_info = "Vital Signs:\n" + "\n".join([
                f"• {key}: {value}" for key, value in vital_signs.items()
            ])
        
        emergency_protocol = ""
        if emergency_info:
            emergency_protocol = "Emergency Protocol:\n" + "\n".join([
                f"• {item['content'][:200]}..." for item in emergency_info[:2]
            ])
        
        return response_template.format(
            symptoms=query,
            vital_signs_info=vital_signs_info,
            emergency_protocol=emergency_protocol
        )
    
    def generate_emergency_recommendations(self, emergency_level: int, 
                                         vital_signs: Dict[str, Any]) -> List[str]:
        """Generate emergency-specific recommendations"""
        recommendations = []
        
        if emergency_level == 5:
            recommendations.extend([
                "Call 911 immediately",
                "Do not attempt to drive yourself",
                "Stay with the patient until help arrives",
                "Follow emergency operator instructions"
            ])
        elif emergency_level == 4:
            recommendations.extend([
                "Go to nearest emergency room",
                "Bring list of current medications",
                "Have someone accompany you if possible",
                "Monitor symptoms during transport"
            ])
        else:
            recommendations.extend([
                "Contact healthcare provider within 24 hours",
                "Monitor symptoms for changes",
                "Keep emergency contacts readily available",
                "Document symptoms and timeline"
            ])
        
        # Add vital signs specific recommendations
        if vital_signs:
            if "temperature" in vital_signs and vital_signs["temperature"] > 103:
                recommendations.append("Apply cool compresses for fever")
            if "blood_pressure" in vital_signs:
                bp = vital_signs["blood_pressure"]
                if bp["systolic"] > 180:
                    recommendations.append("Avoid strenuous activity")
        
        return recommendations
    
    def generate_emergency_questions(self, query: str) -> List[str]:
        """Generate emergency assessment questions"""
        questions = [
            "When did the symptoms start?",
            "Have the symptoms gotten worse?",
            "Are you currently taking any medications?",
            "Do you have any known medical conditions?"
        ]
        
        # Add specific questions based on symptoms
        if "pain" in query.lower():
            questions.extend([
                "On a scale of 1-10, how severe is the pain?",
                "Is the pain constant or intermittent?",
                "What makes the pain better or worse?"
            ])
        
        if "breathing" in query.lower():
            questions.extend([
                "Are you having difficulty breathing at rest or with activity?",
                "Do you have any chest tightness?",
                "Are you using any breathing treatments?"
            ])
        
        return questions

# Additional utility functions for medical agents
def create_medical_agent_factory(rag_system):
    """Factory function to create medical agents"""
    return {
        "symptom": SymptomAnalysisAgent(rag_system),
        "medication": MedicationInformationAgent(rag_system),
        "treatment": TreatmentRecommendationAgent(rag_system),
        "emergency": EmergencyAssessmentAgent(rag_system)
    }

def get_agent_by_query_type(query_type: str, rag_system) -> MedicalAgent:
    """Get appropriate agent based on query type"""
    agents = create_medical_agent_factory(rag_system)
    return agents.get(query_type, agents["symptom"]) 