import pytest
from unittest.mock import MagicMock, patch

from src.models.generator import ContentGenerator
from src.utils.validation import validate_input, validate_model_parameters

@pytest.fixture
def sample_customer_data():
    """Sample customer data for testing."""
    return {
        "customer_id": "12345",
        "name": "John Doe",
        "preferences": {
            "favorite_category": "electronics",
            "price_range": "medium"
        },
        "purchase_history": ["laptop", "smartphone"],
        "demographics": {
            "age": "30",
            "location": "New York"
        }
    }

@pytest.fixture
def generator():
    """Create a ContentGenerator instance for testing."""
    return ContentGenerator(
        model_name="gpt2",
        device="cpu",
        max_length=100,
        temperature=0.7
    )

def test_validate_input_valid_data(sample_customer_data):
    """Test input validation with valid data."""
    validate_input(sample_customer_data, "email")
    # Should not raise any exceptions

def test_validate_input_invalid_content_type(sample_customer_data):
    """Test input validation with invalid content type."""
    with pytest.raises(ValueError):
        validate_input(sample_customer_data, "invalid_type")

def test_validate_input_missing_required_fields():
    """Test input validation with missing required fields."""
    invalid_data = {
        "customer_id": "12345",
        "name": "John Doe"
        # Missing preferences field
    }
    with pytest.raises(ValueError):
        validate_input(invalid_data, "email")

def test_validate_model_parameters_valid():
    """Test model parameter validation with valid parameters."""
    validate_model_parameters(max_length=100, temperature=0.7, num_return_sequences=1)
    # Should not raise any exceptions

def test_validate_model_parameters_invalid():
    """Test model parameter validation with invalid parameters."""
    with pytest.raises(ValueError):
        validate_model_parameters(max_length=5, temperature=0.7, num_return_sequences=1)

@patch('src.models.generator.AutoModelForCausalLM')
@patch('src.models.generator.AutoTokenizer')
def test_generator_initialization(mock_tokenizer, mock_model):
    """Test ContentGenerator initialization."""
    generator = ContentGenerator()
    assert generator.device in ['cuda', 'cpu']
    assert generator.max_length == 100
    assert generator.temperature == 0.7
    mock_model.from_pretrained.assert_called_once()
    mock_tokenizer.from_pretrained.assert_called_once()

@patch('src.models.generator.AutoModelForCausalLM')
@patch('src.models.generator.AutoTokenizer')
def test_generate_content(mock_tokenizer, mock_model, generator, sample_customer_data):
    """Test content generation."""
    # Mock the model's generate method
    mock_output = MagicMock()
    mock_output.tolist.return_value = [[1, 2, 3]]
    mock_model.return_value.generate.return_value = mock_output
    
    # Mock the tokenizer's decode method
    mock_tokenizer.return_value.decode.return_value = "Generated content"
    
    # Test content generation
    content = generator.generate_content(
        customer_data=sample_customer_data,
        content_type="email",
        num_return_sequences=1
    )
    
    assert isinstance(content, list)
    assert len(content) == 1
    assert content[0] == "Generated content"

def test_save_model(generator, tmp_path):
    """Test model saving."""
    save_path = tmp_path / "model"
    generator.save_model(str(save_path))
    # Check if the model files were created
    assert save_path.exists()

@patch('src.models.generator.AutoModelForCausalLM')
@patch('src.models.generator.AutoTokenizer')
def test_load_model(mock_tokenizer, mock_model, tmp_path):
    """Test model loading."""
    # Create a dummy model file
    model_path = tmp_path / "model"
    model_path.mkdir()
    
    # Test loading the model
    generator = ContentGenerator.load_model(str(model_path))
    assert isinstance(generator, ContentGenerator)
    mock_model.from_pretrained.assert_called_once_with(str(model_path))
    mock_tokenizer.from_pretrained.assert_called_once_with(str(model_path))

def test_generator_error_handling(generator, sample_customer_data):
    """Test error handling in content generation."""
    # Test with invalid content type
    with pytest.raises(ValueError):
        generator.generate_content(
            customer_data=sample_customer_data,
            content_type="invalid_type"
        )
    
    # Test with invalid customer data
    invalid_data = {"customer_id": "12345"}  # Missing required fields
    with pytest.raises(ValueError):
        generator.generate_content(
            customer_data=invalid_data,
            content_type="email"
        ) 