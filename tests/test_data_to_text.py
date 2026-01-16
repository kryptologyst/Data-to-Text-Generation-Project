"""
Unit tests for the data-to-text generation project.
"""

import pytest
import json
from unittest.mock import Mock, patch
from pathlib import Path

from src.data_to_text_generator import (
    DataToTextGenerator, 
    DataToTextConfig, 
    StructuredData
)
from src.config import Config, ModelConfig, DataConfig
from src.dataset_generator import SyntheticDatasetGenerator, DatasetConfig


class TestDataToTextConfig:
    """Test the DataToTextConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DataToTextConfig()
        assert config.model_name == "google/flan-t5-base"
        assert config.max_length == 512
        assert config.num_beams == 4
        assert config.temperature == 0.7
        assert config.do_sample is True
        assert config.early_stopping is True
        assert config.device == "auto"
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = DataToTextConfig(
            model_name="t5-small",
            max_length=256,
            temperature=0.5
        )
        assert config.model_name == "t5-small"
        assert config.max_length == 256
        assert config.temperature == 0.5


class TestStructuredData:
    """Test the StructuredData class."""
    
    def test_basic_data(self):
        """Test basic structured data creation."""
        data = {"city": "New York", "temperature": "22°C"}
        structured_data = StructuredData(data=data)
        
        assert structured_data.data == data
        assert structured_data.template is None
        assert structured_data.context is None
    
    def test_data_with_context(self):
        """Test structured data with context."""
        data = {"city": "London", "temperature": "15°C"}
        context = "weather forecast"
        
        structured_data = StructuredData(data=data, context=context)
        
        assert structured_data.data == data
        assert structured_data.context == context


class TestDataToTextGenerator:
    """Test the DataToTextGenerator class."""
    
    @patch('src.data_to_text_generator.AutoTokenizer')
    @patch('src.data_to_text_generator.AutoModelForSeq2SeqLM')
    @patch('src.data_to_text_generator.pipeline')
    def test_initialization(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test generator initialization."""
        config = DataToTextConfig(model_name="t5-small")
        
        # Mock the model loading
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        mock_pipeline.return_value = Mock()
        
        generator = DataToTextGenerator(config)
        
        assert generator.config == config
        assert generator.model is not None
        assert generator.tokenizer is not None
        assert generator.pipeline is not None
    
    def test_format_weather_data(self):
        """Test weather data formatting."""
        config = DataToTextConfig()
        
        with patch('src.data_to_text_generator.AutoTokenizer'), \
             patch('src.data_to_text_generator.AutoModelForSeq2SeqLM'), \
             patch('src.data_to_text_generator.pipeline'):
            
            generator = DataToTextGenerator(config)
            
            weather_data = {
                'city': 'New York',
                'temperature': '22°C',
                'humidity': '60%',
                'forecast': 'sunny',
                'wind_speed': '10 km/h'
            }
            
            structured_data = StructuredData(data=weather_data)
            prompt = generator.format_data_as_prompt(structured_data)
            
            assert "New York" in prompt
            assert "22°C" in prompt
            assert "sunny" in prompt
    
    def test_format_generic_data(self):
        """Test generic data formatting."""
        config = DataToTextConfig()
        
        with patch('src.data_to_text_generator.AutoTokenizer'), \
             patch('src.data_to_text_generator.AutoModelForSeq2SeqLM'), \
             patch('src.data_to_text_generator.pipeline'):
            
            generator = DataToTextGenerator(config)
            
            data = {"name": "John", "age": "30"}
            context = "person information"
            
            structured_data = StructuredData(data=data, context=context)
            prompt = generator.format_data_as_prompt(structured_data)
            
            assert "person information" in prompt
            assert "name: John" in prompt
            assert "age: 30" in prompt


class TestSyntheticDatasetGenerator:
    """Test the SyntheticDatasetGenerator class."""
    
    def test_initialization(self):
        """Test dataset generator initialization."""
        config = DatasetConfig(num_samples=100)
        generator = SyntheticDatasetGenerator(config)
        
        assert generator.config == config
        assert generator.fake is not None
    
    def test_generate_weather_data(self):
        """Test weather data generation."""
        config = DatasetConfig(num_samples=5)
        generator = SyntheticDatasetGenerator(config)
        
        data = generator.generate_weather_data(5)
        
        assert len(data) == 5
        for item in data:
            assert "city" in item
            assert "temperature" in item
            assert "humidity" in item
            assert "forecast" in item
            assert "wind_speed" in item
            assert item["category"] == "weather"
    
    def test_generate_product_data(self):
        """Test product data generation."""
        config = DatasetConfig(num_samples=5)
        generator = SyntheticDatasetGenerator(config)
        
        data = generator.generate_product_data(5)
        
        assert len(data) == 5
        for item in data:
            assert "product" in item
            assert "price" in item
            assert "storage" in item
            assert "color" in item
            assert "availability" in item
            assert item["category"] == "product"
    
    def test_create_reference_texts(self):
        """Test reference text creation."""
        config = DatasetConfig()
        generator = SyntheticDatasetGenerator(config)
        
        data = [
            {
                "city": "New York",
                "temperature": "22°C",
                "humidity": "60%",
                "forecast": "sunny",
                "wind_speed": "10 km/h",
                "category": "weather"
            }
        ]
        
        reference_texts = generator.create_reference_texts(data)
        
        assert len(reference_texts) == 1
        assert "New York" in reference_texts[0]
        assert "sunny" in reference_texts[0]


class TestConfig:
    """Test the Config class."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = Config(
            model=ModelConfig(),
            data=DataConfig(),
            evaluation=Mock(),
            app=Mock()
        )
        
        assert config.model.model_name == "google/flan-t5-base"
        assert config.data.batch_size == 8
        assert config.data.random_seed == 42


def test_create_sample_dataset():
    """Test sample dataset creation."""
    from src.data_to_text_generator import create_sample_dataset
    
    dataset = create_sample_dataset()
    
    assert len(dataset) == 4
    assert all(isinstance(item, StructuredData) for item in dataset)
    
    # Check that we have different categories
    contexts = [item.context for item in dataset]
    assert "weather forecast" in contexts
    assert "product information" in contexts


if __name__ == "__main__":
    pytest.main([__file__])
