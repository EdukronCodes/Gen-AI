from typing import Dict, Union

def validate_input(
    customer_data: Dict[str, Union[str, int, float]],
    content_type: str
) -> None:
    """
    Validate input data for content generation.

    Args:
        customer_data: Dictionary containing customer information
        content_type: Type of content to generate

    Raises:
        ValueError: If input validation fails
    """
    # Validate content type
    valid_content_types = {'email', 'social_media', 'banner', 'notification'}
    if content_type not in valid_content_types:
        raise ValueError(
            f"Invalid content type: {content_type}. "
            f"Must be one of {valid_content_types}"
        )

    # Validate customer data
    if not customer_data:
        raise ValueError("Customer data cannot be empty")

    # Check required fields
    required_fields = {'customer_id', 'name', 'preferences'}
    missing_fields = required_fields - set(customer_data.keys())
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    # Validate field types
    for field, value in customer_data.items():
        if not isinstance(value, (str, int, float)):
            raise ValueError(
                f"Invalid type for field '{field}': {type(value)}. "
                "Must be string, integer, or float"
            )

def validate_model_parameters(
    max_length: int,
    temperature: float,
    num_return_sequences: int
) -> None:
    """
    Validate model generation parameters.

    Args:
        max_length: Maximum length of generated content
        temperature: Sampling temperature
        num_return_sequences: Number of sequences to generate

    Raises:
        ValueError: If parameter validation fails
    """
    if max_length < 10 or max_length > 1000:
        raise ValueError("max_length must be between 10 and 1000")

    if temperature < 0.1 or temperature > 2.0:
        raise ValueError("temperature must be between 0.1 and 2.0")

    if num_return_sequences < 1 or num_return_sequences > 10:
        raise ValueError("num_return_sequences must be between 1 and 10")

def validate_training_data(training_data: list) -> None:
    """
    Validate training data format and content.

    Args:
        training_data: List of training examples

    Raises:
        ValueError: If training data validation fails
    """
    if not training_data:
        raise ValueError("Training data cannot be empty")

    for example in training_data:
        if not isinstance(example, dict):
            raise ValueError("Each training example must be a dictionary")

        required_keys = {'input', 'output'}
        missing_keys = required_keys - set(example.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys in training example: {missing_keys}")

        if not isinstance(example['input'], str) or not isinstance(example['output'], str):
            raise ValueError("Input and output must be strings")

def validate_model_path(path: str) -> None:
    """
    Validate model save/load path.

    Args:
        path: Path to validate

    Raises:
        ValueError: If path validation fails
    """
    if not path:
        raise ValueError("Model path cannot be empty")

    if not isinstance(path, str):
        raise ValueError("Model path must be a string")

    # Check for invalid characters
    invalid_chars = {'<', '>', ':', '"', '/', '\\', '|', '?', '*'}
    if any(char in path for char in invalid_chars):
        raise ValueError(f"Model path contains invalid characters: {invalid_chars}") 