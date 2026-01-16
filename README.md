# Data-to-Text Generation Project

## Overview
This project provides a production-ready implementation for converting structured data into coherent, human-readable text using state-of-the-art transformer models.

## Features
- **Multiple Models**: Support for FLAN-T5, T5, and other transformer models
- **Type Safety**: Full type hints and modern Python practices
- **Web Interface**: Streamlit-based UI for easy interaction
- **Batch Processing**: Handle multiple data items efficiently
- **Synthetic Data**: Generate datasets for training and evaluation
- **Evaluation**: ROUGE metrics for model assessment
- **Configuration**: YAML/JSON configuration management
- **Logging**: Comprehensive logging and monitoring

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster inference)

### Setup
```bash
# Clone the repository
git clone https://github.com/kryptologyst/Data-to-Text-Generation-Project.git
cd Data-to-Text-Generation-Project

# Install dependencies
pip install -r requirements.txt

# Download models (optional, will download automatically on first use)
python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; AutoTokenizer.from_pretrained('google/flan-t5-base')"
```

## Quick Start

### Command Line Usage
```python
from src.data_to_text_generator import DataToTextGenerator, DataToTextConfig, StructuredData

# Initialize generator
config = DataToTextConfig(model_name="google/flan-t5-base")
generator = DataToTextGenerator(config)

# Create structured data
data = {
    'city': 'New York',
    'temperature': '22°C',
    'humidity': '60%',
    'forecast': 'sunny',
    'wind_speed': '10 km/h'
}

structured_data = StructuredData(data=data, context="weather forecast")

# Generate text
result = generator.generate_text(structured_data)
print(result['generated_text'])
```

### Web Interface
```bash
# Start the Streamlit app
streamlit run web_app/app.py
```

Then open your browser to `http://localhost:8501`

## Project Structure
```
├── src/                          # Source code
│   ├── data_to_text_generator.py # Main generator class
│   ├── config.py                 # Configuration management
│   └── dataset_generator.py      # Synthetic dataset generation
├── web_app/                      # Web interface
│   └── app.py                    # Streamlit application
├── config/                       # Configuration files
│   └── config.yaml               # Default configuration
├── data/                         # Data storage
├── models/                       # Saved models
├── tests/                        # Unit tests
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## Configuration

The project uses YAML configuration files. See `config/config.yaml` for available options:

- **Model settings**: Model name, generation parameters
- **Data settings**: Batch size, validation split
- **Evaluation settings**: Metrics, output directory
- **App settings**: Title, debug mode, logging level

## Usage Examples

### Weather Data
```python
weather_data = {
    'city': 'London',
    'temperature': '15°C',
    'humidity': '80%',
    'forecast': 'cloudy',
    'wind_speed': '15 km/h'
}

structured_data = StructuredData(data=weather_data, context="weather forecast")
result = generator.generate_text(structured_data)
```

### Product Information
```python
product_data = {
    'product': 'iPhone 15',
    'price': '$999',
    'storage': '128GB',
    'color': 'Space Black',
    'availability': 'In Stock'
}

structured_data = StructuredData(data=product_data, context="product information")
result = generator.generate_text(structured_data)
```

### Batch Processing
```python
data_list = [weather_data, product_data]
structured_data_list = [StructuredData(data=d) for d in data_list]
results = generator.batch_generate(structured_data_list)
```

## Synthetic Dataset Generation

Generate synthetic datasets for training and evaluation:

```python
from src.dataset_generator import SyntheticDatasetGenerator, DatasetConfig

config = DatasetConfig(
    num_samples=1000,
    categories=["weather", "product", "company"]
)

generator = SyntheticDatasetGenerator(config)
data = generator.generate_dataset()
reference_texts = generator.create_reference_texts(data)
```

## Model Evaluation

Evaluate model performance using ROUGE metrics:

```python
generated_texts = ["Generated text 1", "Generated text 2"]
reference_texts = ["Reference text 1", "Reference text 2"]

metrics = generator.evaluate_generation(generated_texts, reference_texts)
print(f"ROUGE-1: {metrics['rouge1']:.4f}")
print(f"ROUGE-2: {metrics['rouge2']:.4f}")
```

## Available Models

- **google/flan-t5-base**: Balanced performance and speed
- **google/flan-t5-small**: Faster, smaller model
- **google/flan-t5-large**: Better quality, slower
- **t5-base**: Original T5 model
- **t5-small**: Smaller T5 model

## Performance Tips

1. **GPU Usage**: Use CUDA for faster inference
2. **Batch Processing**: Process multiple items together
3. **Model Selection**: Choose appropriate model size for your needs
4. **Temperature**: Lower values (0.1-0.5) for more focused output
5. **Max Length**: Adjust based on expected output length

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Quality
```bash
# Format code
black src/ web_app/ tests/

# Lint code
flake8 src/ web_app/ tests/

# Type checking
mypy src/
```

### Adding New Models
1. Add model name to `ModelConfig`
2. Update model options in web interface
3. Test with sample data
4. Update documentation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face Transformers library
- FLAN-T5 model by Google
- Streamlit for the web interface
- The open-source AI community
# Data-to-Text-Generation-Project
