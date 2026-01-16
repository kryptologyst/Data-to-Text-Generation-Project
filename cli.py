#!/usr/bin/env python3
"""
Command-line interface for data-to-text generation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_to_text_generator import DataToTextGenerator, DataToTextConfig, StructuredData
from dataset_generator import SyntheticDatasetGenerator, DatasetConfig
from config import load_config


def generate_text_cli(args):
    """Generate text from structured data."""
    # Load configuration
    config = load_config(args.config)
    
    # Create generator
    generator_config = DataToTextConfig(
        model_name=args.model,
        max_length=args.max_length,
        temperature=args.temperature,
        num_beams=args.num_beams
    )
    
    generator = DataToTextGenerator(generator_config)
    
    # Load data
    if args.input:
        with open(args.input, 'r') as f:
            data = json.load(f)
    else:
        # Use sample data
        data = {
            'city': 'New York',
            'temperature': '22°C',
            'humidity': '60%',
            'forecast': 'sunny',
            'wind_speed': '10 km/h'
        }
    
    # Generate text
    structured_data = StructuredData(data=data, context=args.context)
    result = generator.generate_text(structured_data)
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print("Generated Text:")
        print(result['generated_text'])
        
        if args.verbose:
            print("\nDetails:")
            print(f"Model: {result['model']}")
            print(f"Device: {result['device']}")
            print(f"Prompt: {result['prompt']}")


def generate_dataset_cli(args):
    """Generate synthetic dataset."""
    config = DatasetConfig(
        num_samples=args.samples,
        categories=args.categories.split(',') if args.categories else None,
        random_seed=args.seed,
        output_dir=args.output_dir
    )
    
    generator = SyntheticDatasetGenerator(config)
    
    print(f"Generating {args.samples} samples...")
    data = generator.generate_dataset()
    reference_texts = generator.create_reference_texts(data)
    
    generator.save_dataset(data, reference_texts)
    
    print(f"Dataset generated successfully!")
    print(f"Categories: {', '.join(set(item['category'] for item in data))}")


def evaluate_cli(args):
    """Evaluate model performance."""
    # Load configuration
    config = load_config(args.config)
    
    # Create generator
    generator_config = DataToTextConfig(model_name=args.model)
    generator = DataToTextGenerator(generator_config)
    
    # Load generated and reference texts
    with open(args.generated, 'r') as f:
        generated_texts = [line.strip() for line in f if line.strip()]
    
    with open(args.reference, 'r') as f:
        reference_texts = [line.strip() for line in f if line.strip()]
    
    if len(generated_texts) != len(reference_texts):
        print("Error: Number of generated and reference texts must match")
        sys.exit(1)
    
    # Evaluate
    metrics = generator.evaluate_generation(generated_texts, reference_texts)
    
    # Output results
    print("Evaluation Results:")
    print("=" * 30)
    for metric, score in metrics.items():
        print(f"{metric.upper()}: {score:.4f}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {args.output}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Data-to-Text Generation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate text from sample data
  python cli.py generate

  # Generate text from custom data file
  python cli.py generate --input data.json --output result.json

  # Generate synthetic dataset
  python cli.py dataset --samples 1000 --categories weather,product

  # Evaluate model performance
  python cli.py evaluate --generated generated.txt --reference reference.txt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate text from structured data')
    generate_parser.add_argument('--input', '-i', help='Input JSON file with data')
    generate_parser.add_argument('--output', '-o', help='Output JSON file')
    generate_parser.add_argument('--model', '-m', default='google/flan-t5-base', help='Model name')
    generate_parser.add_argument('--max-length', type=int, default=256, help='Maximum output length')
    generate_parser.add_argument('--temperature', type=float, default=0.7, help='Generation temperature')
    generate_parser.add_argument('--num-beams', type=int, default=4, help='Number of beams')
    generate_parser.add_argument('--context', help='Context for the data')
    generate_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    generate_parser.add_argument('--config', help='Configuration file')
    
    # Dataset command
    dataset_parser = subparsers.add_parser('dataset', help='Generate synthetic dataset')
    dataset_parser.add_argument('--samples', '-n', type=int, default=1000, help='Number of samples')
    dataset_parser.add_argument('--categories', help='Comma-separated categories')
    dataset_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    dataset_parser.add_argument('--output-dir', default='data', help='Output directory')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    eval_parser.add_argument('--generated', '-g', required=True, help='File with generated texts')
    eval_parser.add_argument('--reference', '-r', required=True, help='File with reference texts')
    eval_parser.add_argument('--model', '-m', default='google/flan-t5-base', help='Model name')
    eval_parser.add_argument('--output', '-o', help='Output file for metrics')
    eval_parser.add_argument('--config', help='Configuration file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'generate':
            generate_text_cli(args)
        elif args.command == 'dataset':
            generate_dataset_cli(args)
        elif args.command == 'evaluate':
            evaluate_cli(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
