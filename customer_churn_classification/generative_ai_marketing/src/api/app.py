import logging
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..models.generator import ContentGenerator
from ..utils.validation import validate_input, validate_model_parameters

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Personalized Marketing Content Generator",
    description="API for generating personalized marketing content using AI",
    version="1.0.0"
)

# Initialize content generator
generator = ContentGenerator()

class CustomerData(BaseModel):
    """Customer data model for content generation."""
    customer_id: str = Field(..., description="Unique customer identifier")
    name: str = Field(..., description="Customer name")
    preferences: Dict[str, str] = Field(..., description="Customer preferences")
    purchase_history: Optional[List[str]] = Field(None, description="List of purchased items")
    demographics: Optional[Dict[str, str]] = Field(None, description="Customer demographics")

class ContentRequest(BaseModel):
    """Request model for content generation."""
    customer_data: CustomerData
    content_type: str = Field(..., description="Type of content to generate")
    num_variations: int = Field(1, ge=1, le=10, description="Number of content variations")

class ContentResponse(BaseModel):
    """Response model for generated content."""
    content: List[str]
    metadata: Dict[str, str]

@app.post("/generate", response_model=ContentResponse)
async def generate_content(request: ContentRequest):
    """
    Generate personalized marketing content.

    Args:
        request: Content generation request

    Returns:
        Generated content and metadata
    """
    try:
        # Validate input
        validate_input(request.customer_data.dict(), request.content_type)
        validate_model_parameters(
            max_length=100,
            temperature=0.7,
            num_return_sequences=request.num_variations
        )

        # Generate content
        content = generator.generate_content(
            customer_data=request.customer_data.dict(),
            content_type=request.content_type,
            num_return_sequences=request.num_variations
        )

        # Prepare response
        response = ContentResponse(
            content=content,
            metadata={
                "content_type": request.content_type,
                "customer_id": request.customer_data.customer_id,
                "num_variations": str(request.num_variations)
            }
        )

        return response

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating content: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model."""
    return {
        "model_name": generator.model.name_or_path,
        "device": generator.device,
        "max_length": generator.max_length,
        "temperature": generator.temperature
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 