"""
Data-to-Text Generation Module

This module provides a modern, type-safe implementation for converting structured data
into coherent, human-readable text using state-of-the-art transformer models.
"""

import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import json
import yaml

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    pipeline,
    Pipeline
)
from datasets import Dataset
import pandas as pd
from sklearn.metrics import rouge_score
import evaluate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataToTextConfig:
    """Configuration class for data-to-text generation."""
    model_name: str = "google/flan-t5-base"
    max_length: int = 512
    num_beams: int = 4
    temperature: float = 0.7
    do_sample: bool = True
    early_stopping: bool = True
    device: str = "auto"


@dataclass
class StructuredData:
    """Represents structured data for text generation."""
    data: Dict[str, Any]
    template: Optional[str] = None
    context: Optional[str] = None


class DataToTextGenerator:
    """
    Modern data-to-text generation using transformer models.
    
    Supports multiple models, templates, and evaluation metrics.
    """
    
    def __init__(self, config: DataToTextConfig):
        """
        Initialize the data-to-text generator.
        
        Args:
            config: Configuration object containing model and generation parameters
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = self._get_device()
        
        logger.info(f"Initializing DataToTextGenerator with model: {config.model_name}")
        self._load_model()
    
    def _get_device(self) -> str:
        """Determine the best available device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.config.device
    
    def _load_model(self) -> None:
        """Load the transformer model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.config.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Create pipeline for easier generation
            self.pipeline = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            logger.info(f"Model loaded successfully on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def format_data_as_prompt(self, structured_data: StructuredData) -> str:
        """
        Convert structured data into a text prompt.
        
        Args:
            structured_data: The structured data to convert
            
        Returns:
            Formatted prompt string
        """
        if structured_data.template:
            return structured_data.template.format(**structured_data.data)
        
        # Default template for weather data
        if 'city' in structured_data.data and 'temperature' in structured_data.data:
            return self._format_weather_data(structured_data.data)
        
        # Generic template for other data types
        return self._format_generic_data(structured_data.data, structured_data.context)
    
    def _format_weather_data(self, data: Dict[str, Any]) -> str:
        """Format weather data into a natural language prompt."""
        template = (
            "Generate a weather report for {city}. "
            "Temperature: {temperature}, Humidity: {humidity}, "
            "Conditions: {forecast}, Wind: {wind_speed}"
        )
        return template.format(**data)
    
    def _format_generic_data(self, data: Dict[str, Any], context: Optional[str] = None) -> str:
        """Format generic structured data into a prompt."""
        prompt_parts = []
        
        if context:
            prompt_parts.append(f"Context: {context}")
        
        prompt_parts.append("Convert the following data into natural language:")
        
        for key, value in data.items():
            prompt_parts.append(f"{key}: {value}")
        
        return " ".join(prompt_parts)
    
    def generate_text(
        self, 
        structured_data: StructuredData,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Generate text from structured data.
        
        Args:
            structured_data: The structured data to convert
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generated text and metadata
        """
        try:
            prompt = self.format_data_as_prompt(structured_data)
            
            # Merge generation parameters
            gen_params = {
                "max_length": self.config.max_length,
                "num_beams": self.config.num_beams,
                "temperature": self.config.temperature,
                "do_sample": self.config.do_sample,
                "early_stopping": self.config.early_stopping,
                **generation_kwargs
            }
            
            logger.info(f"Generating text for prompt: {prompt[:100]}...")
            
            # Generate text
            result = self.pipeline(prompt, **gen_params)
            
            generated_text = result[0]['generated_text']
            
            return {
                "generated_text": generated_text,
                "prompt": prompt,
                "model": self.config.model_name,
                "generation_params": gen_params,
                "device": self.device
            }
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    def batch_generate(
        self, 
        structured_data_list: List[StructuredData],
        **generation_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate text for multiple structured data inputs.
        
        Args:
            structured_data_list: List of structured data objects
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of generation results
        """
        results = []
        
        for i, structured_data in enumerate(structured_data_list):
            logger.info(f"Processing item {i+1}/{len(structured_data_list)}")
            result = self.generate_text(structured_data, **generation_kwargs)
            results.append(result)
        
        return results
    
    def evaluate_generation(
        self, 
        generated_texts: List[str], 
        reference_texts: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate generated texts against reference texts using ROUGE metrics.
        
        Args:
            generated_texts: List of generated texts
            reference_texts: List of reference texts
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            rouge = evaluate.load('rouge')
            results = rouge.compute(
                predictions=generated_texts,
                references=reference_texts
            )
            
            return {
                "rouge1": results["rouge1"],
                "rouge2": results["rouge2"],
                "rougeL": results["rougeL"],
                "rougeLsum": results["rougeLsum"]
            }
            
        except Exception as e:
            logger.error(f"Error evaluating generation: {e}")
            return {}
    
    def save_model(self, path: Union[str, Path]) -> None:
        """Save the model and tokenizer to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save config
        config_path = path / "generator_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: Union[str, Path]) -> 'DataToTextGenerator':
        """Load a saved model and configuration."""
        path = Path(path)
        
        # Load config
        config_path = path / "generator_config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = DataToTextConfig(**config_dict)
        
        # Create instance and load model
        instance = cls(config)
        instance.model = AutoModelForSeq2SeqLM.from_pretrained(path)
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        
        return instance


def create_sample_dataset() -> List[StructuredData]:
    """Create a sample dataset for demonstration purposes."""
    sample_data = [
        StructuredData(
            data={
                'city': 'New York',
                'temperature': '22°C',
                'humidity': '60%',
                'forecast': 'sunny',
                'wind_speed': '10 km/h'
            },
            context="weather forecast"
        ),
        StructuredData(
            data={
                'city': 'London',
                'temperature': '15°C',
                'humidity': '80%',
                'forecast': 'cloudy',
                'wind_speed': '15 km/h'
            },
            context="weather forecast"
        ),
        StructuredData(
            data={
                'product': 'iPhone 15',
                'price': '$999',
                'storage': '128GB',
                'color': 'Space Black',
                'availability': 'In Stock'
            },
            context="product information"
        ),
        StructuredData(
            data={
                'company': 'TechCorp',
                'revenue': '$2.5M',
                'employees': '150',
                'industry': 'Software',
                'location': 'San Francisco'
            },
            context="company profile"
        )
    ]
    
    return sample_data


def main():
    """Main function for demonstration."""
    # Create configuration
    config = DataToTextConfig(
        model_name="google/flan-t5-base",
        max_length=256,
        temperature=0.7
    )
    
    # Initialize generator
    generator = DataToTextGenerator(config)
    
    # Create sample data
    sample_data = create_sample_dataset()
    
    # Generate text for each sample
    print("Data-to-Text Generation Results:")
    print("=" * 50)
    
    for i, structured_data in enumerate(sample_data, 1):
        print(f"\nSample {i}:")
        print(f"Input Data: {structured_data.data}")
        
        result = generator.generate_text(structured_data)
        print(f"Generated Text: {result['generated_text']}")
        print("-" * 30)


if __name__ == "__main__":
    main()
