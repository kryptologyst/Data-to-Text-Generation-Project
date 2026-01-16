"""
Configuration management for the data-to-text generation project.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    name: str = "google/flan-t5-base"
    max_length: int = 512
    num_beams: int = 4
    temperature: float = 0.7
    do_sample: bool = True
    early_stopping: bool = True
    device: str = "auto"


@dataclass
class DataConfig:
    """Data processing configuration."""
    batch_size: int = 8
    max_samples: Optional[int] = None
    validation_split: float = 0.2
    random_seed: int = 42


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    metrics: list = None
    save_predictions: bool = True
    output_dir: str = "results"
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["rouge1", "rouge2", "rougeL", "rougeLsum"]


@dataclass
class AppConfig:
    """Application configuration."""
    title: str = "Data-to-Text Generation"
    description: str = "Convert structured data into natural language"
    debug: bool = False
    log_level: str = "INFO"


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig
    data: DataConfig
    evaluation: EvaluationConfig
    app: AppConfig
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
            app=AppConfig(**config_dict.get('app', {}))
        )
    
    @classmethod
    def from_json(cls, config_path: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
            app=AppConfig(**config_dict.get('app', {}))
        )
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'model': asdict(self.model),
            'data': asdict(self.data),
            'evaluation': asdict(self.evaluation),
            'app': asdict(self.app)
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def to_json(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = {
            'model': asdict(self.model),
            'data': asdict(self.data),
            'evaluation': asdict(self.evaluation),
            'app': asdict(self.app)
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


def create_default_config() -> Config:
    """Create default configuration."""
    return Config(
        model=ModelConfig(),
        data=DataConfig(),
        evaluation=EvaluationConfig(),
        app=AppConfig()
   )


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or create default."""
    if config_path and Path(config_path).exists():
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return Config.from_yaml(config_path)
        elif config_path.endswith('.json'):
            return Config.from_json(config_path)
        else:
            raise ValueError("Config file must be .yaml, .yml, or .json")
    
    return create_default_config()
