import re
from typing import Dict, Union

def clean_text(text: str) -> str:
    """
    Clean and normalize generated text.

    Args:
        text: Input text to clean

    Returns:
        Cleaned text string
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    # Fix common spacing issues
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    
    # Capitalize first letter
    text = text.strip().capitalize()
    
    return text

def format_prompt(
    customer_data: Dict[str, Union[str, int, float]],
    content_type: str
) -> str:
    """
    Format the prompt for content generation based on customer data and content type.

    Args:
        customer_data: Dictionary containing customer information
        content_type: Type of content to generate

    Returns:
        Formatted prompt string
    """
    # Base prompt template
    base_prompt = (
        "Generate a personalized {content_type} marketing message for a customer "
        "with the following characteristics:\n"
    )

    # Add customer data to prompt
    customer_info = []
    for key, value in customer_data.items():
        if isinstance(value, (int, float)):
            customer_info.append(f"{key}: {value}")
        else:
            customer_info.append(f"{key}: '{value}'")

    # Combine all parts
    prompt = base_prompt.format(content_type=content_type)
    prompt += "\n".join(customer_info)
    prompt += "\n\nGenerate a personalized message:"

    return prompt

def extract_keywords(text: str) -> list:
    """
    Extract important keywords from text.

    Args:
        text: Input text to process

    Returns:
        List of extracted keywords
    """
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    # Convert to lowercase and split
    words = text.lower().split()
    
    # Filter out stop words and short words
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    return keywords

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two text strings.

    Args:
        text1: First text string
        text2: Second text string

    Returns:
        Similarity score between 0 and 1
    """
    # Convert to sets of words
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

def validate_content_length(text: str, min_length: int = 50, max_length: int = 500) -> bool:
    """
    Validate if the content length is within acceptable limits.

    Args:
        text: Content to validate
        min_length: Minimum acceptable length
        max_length: Maximum acceptable length

    Returns:
        Boolean indicating if length is valid
    """
    length = len(text.split())
    return min_length <= length <= max_length 