"""
Data-to-Text Generation Package

A modern, production-ready implementation for converting structured data
into coherent, human-readable text using state-of-the-art transformer models.
"""

from .data_to_text_generator import (
    DataToTextGenerator,
    DataToTextConfig,
    StructuredData,
    create_sample_dataset
)

from .config import (
    Config,
    ModelConfig,
    DataConfig,
    EvaluationConfig,
    AppConfig,
    load_config,
    create_default_config
)

from .dataset_generator import (
    SyntheticDatasetGenerator,
    DatasetConfig
)

__version__ = "1.0.0"
__author__ = "Data-to-Text Generation Team"

__all__ = [
    "DataToTextGenerator",
    "DataToTextConfig", 
    "StructuredData",
    "create_sample_dataset",
    "Config",
    "ModelConfig",
    "DataConfig",
    "EvaluationConfig",
    "AppConfig",
    "load_config",
    "create_default_config",
    "SyntheticDatasetGenerator",
    "DatasetConfig"
]
