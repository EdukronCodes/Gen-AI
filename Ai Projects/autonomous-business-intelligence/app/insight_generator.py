"""
Insight Generator - Generates business insights from analysis results
"""
from typing import Dict, Any, List
from openai import OpenAI
from app.models import Insight, InsightType, InsightSeverity
from config import settings


class InsightGenerator:
    """Generate actionable business insights"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
    
    def generate_insight(
        self,
        analysis_result: Dict[str, Any],
        insight_type: InsightType,
        category: str
    ) -> Insight:
        """Generate insight from analysis result"""
        
        # Determine severity based on insight type and data
        severity = self._determine_severity(analysis_result, insight_type)
        
        # Generate title and description using LLM
        title, description = self._generate_text(analysis_result, insight_type, category)
        
        # Extract recommendations
        recommendations = self._generate_recommendations(analysis_result, insight_type)
        
        # Calculate confidence
        confidence = self._calculate_confidence(analysis_result)
        
        # Extract data points
        data_points = self._extract_data_points(analysis_result)
        
        # Assess business impact
        business_impact = self._assess_business_impact(analysis_result, insight_type)
        
        return Insight(
            type=insight_type,
            title=title,
            description=description,
            category=category,
            severity=severity,
            confidence=confidence,
            data_points=data_points,
            recommendations=recommendations,
            business_impact=business_impact
        )
    
    def _determine_severity(self, result: Dict[str, Any], insight_type: InsightType) -> InsightSeverity:
        """Determine insight severity"""
        if insight_type == InsightType.RISK:
            return InsightSeverity.CRITICAL
        elif insight_type == InsightType.ANOMALY:
            # Check anomaly percentage
            anomaly_pct = result.get("anomaly_percentage", 0)
            if anomaly_pct > 10:
                return InsightSeverity.CRITICAL
            elif anomaly_pct > 5:
                return InsightSeverity.HIGH
            else:
                return InsightSeverity.MEDIUM
        elif insight_type == InsightType.OPPORTUNITY:
            return InsightSeverity.MEDIUM
        else:
            return InsightSeverity.LOW
    
    def _generate_text(self, result: Dict[str, Any], insight_type: InsightType, category: str) -> tuple:
        """Generate title and description using LLM"""
        if not settings.openai_api_key:
            return (
                f"{insight_type.value.title()} in {category}",
                f"Analysis result: {str(result)[:200]}"
            )
        
        try:
            prompt = f"""Generate a concise business insight title and description based on this analysis result.
            
Insight Type: {insight_type.value}
Category: {category}
Analysis Result: {str(result)[:1000]}

Respond in JSON format: {{"title": "...", "description": "..."}}"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a business intelligence analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            import json
            text_data = json.loads(response.choices[0].message.content.strip())
            return text_data.get("title", ""), text_data.get("description", "")
        
        except Exception as e:
            return (
                f"{insight_type.value.title()} in {category}",
                f"Analysis result: {str(result)[:200]}"
            )
    
    def _generate_recommendations(self, result: Dict[str, Any], insight_type: InsightType) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if insight_type == InsightType.ANOMALY:
            recommendations.append("Investigate the identified anomalies to understand root causes")
            recommendations.append("Review data quality and collection processes")
        
        elif insight_type == InsightType.TREND:
            if result.get("trends"):
                for trend_name, trend_data in result["trends"].items():
                    if trend_data.get("trend") == "increasing":
                        recommendations.append(f"Leverage the increasing trend in {trend_name}")
                    elif trend_data.get("trend") == "decreasing":
                        recommendations.append(f"Address the declining trend in {trend_name}")
        
        elif insight_type == InsightType.CORRELATION:
            recommendations.append("Investigate the identified correlations for causal relationships")
            recommendations.append("Consider these relationships in future planning")
        
        elif insight_type == InsightType.PREDICTION:
            recommendations.append("Monitor actual results against predictions")
            recommendations.append("Adjust strategies based on forecasted trends")
        
        return recommendations
    
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate confidence score"""
        # Base confidence
        confidence = 0.7
        
        # Adjust based on data quality
        if "validation" in result:
            validation = result["validation"]
            if validation.get("is_valid", False):
                confidence += 0.1
        
        # Adjust based on statistical significance
        if "p_value" in str(result):
            # Extract p-value if present
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _extract_data_points(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract key data points from result"""
        data_points = []
        
        # Extract key metrics
        if "anomaly_count" in result:
            data_points.append({
                "metric": "anomaly_count",
                "value": result["anomaly_count"]
            })
        
        if "forecast" in result:
            data_points.append({
                "metric": "forecast_horizon",
                "value": result.get("forecast_horizon", 0)
            })
        
        if "trends" in result:
            for trend_name, trend_data in result["trends"].items():
                data_points.append({
                    "metric": f"trend_{trend_name}",
                    "value": trend_data.get("slope", 0)
                })
        
        return data_points
    
    def _assess_business_impact(self, result: Dict[str, Any], insight_type: InsightType) -> Dict[str, Any]:
        """Assess potential business impact"""
        impact = {
            "potential_impact": "medium",
            "affected_areas": [],
            "urgency": "medium"
        }
        
        if insight_type == InsightType.RISK:
            impact["potential_impact"] = "high"
            impact["urgency"] = "high"
        
        elif insight_type == InsightType.ANOMALY:
            anomaly_pct = result.get("anomaly_percentage", 0)
            if anomaly_pct > 10:
                impact["potential_impact"] = "high"
                impact["urgency"] = "high"
        
        elif insight_type == InsightType.OPPORTUNITY:
            impact["potential_impact"] = "positive"
            impact["urgency"] = "medium"
        
        return impact

