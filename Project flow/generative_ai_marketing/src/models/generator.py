import logging
from typing import Dict, List, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

from ..utils.text_utils import clean_text, format_prompt
from ..utils.validation import validate_input

logger = logging.getLogger(__name__)

class ContentGenerator:
    """Generates personalized marketing content using transformer models."""

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 100,
        temperature: float = 0.7,
    ):
        """
        Initialize the content generator.

        Args:
            model_name: Name of the pre-trained model to use
            device: Device to run the model on ('cuda' or 'cpu')
            max_length: Maximum length of generated content
            temperature: Sampling temperature for generation
        """
        self.device = device
        self.max_length = max_length
        self.temperature = temperature

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            logger.info(f"Successfully loaded model {model_name}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def generate_content(
        self,
        customer_data: Dict[str, Union[str, int, float]],
        content_type: str,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate personalized marketing content.

        Args:
            customer_data: Dictionary containing customer information
            content_type: Type of content to generate (e.g., 'email', 'social_media')
            num_return_sequences: Number of different content variations to generate

        Returns:
            List of generated content strings
        """
        # Validate inputs
        validate_input(customer_data, content_type)

        # Format prompt based on content type and customer data
        prompt = format_prompt(customer_data, content_type)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        try:
            # Generate content
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                temperature=self.temperature,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.9,
                top_k=50,
            )

            # Decode and clean generated content
            generated_texts = []
            for output in outputs:
                text = self.tokenizer.decode(output, skip_special_tokens=True)
                cleaned_text = clean_text(text)
                generated_texts.append(cleaned_text)

            return generated_texts

        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            raise

    def fine_tune(
        self,
        training_data: List[Dict[str, str]],
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
    ) -> None:
        """
        Fine-tune the model on custom marketing content.

        Args:
            training_data: List of dictionaries containing training examples
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
        """
        # Implementation for fine-tuning
        pass

    def save_model(self, path: str) -> None:
        """
        Save the model and tokenizer.

        Args:
            path: Directory path to save the model
        """
        try:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            logger.info(f"Model saved successfully to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    @classmethod
    def load_model(cls, path: str, **kwargs) -> "ContentGenerator":
        """
        Load a saved model.

        Args:
            path: Directory path containing the saved model
            **kwargs: Additional arguments for initialization

        Returns:
            ContentGenerator instance with loaded model
        """
        try:
            instance = cls(**kwargs)
            instance.model = AutoModelForCausalLM.from_pretrained(path)
            instance.tokenizer = AutoTokenizer.from_pretrained(path)
            return instance
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise 