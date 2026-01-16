#!/usr/bin/env python3
"""
Demo script for the data-to-text generation project.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_to_text_generator import DataToTextGenerator, DataToTextConfig, StructuredData, create_sample_dataset


def main():
    """Run the demo."""
    print("🚀 Data-to-Text Generation Demo")
    print("=" * 40)
    
    # Create configuration
    config = DataToTextConfig(
        model_name="google/flan-t5-base",
        max_length=256,
        temperature=0.7
    )
    
    print(f"📋 Configuration:")
    print(f"   Model: {config.model_name}")
    print(f"   Max Length: {config.max_length}")
    print(f"   Temperature: {config.temperature}")
    print()
    
    try:
        # Initialize generator
        print("🔄 Loading model...")
        generator = DataToTextGenerator(config)
        print("✅ Model loaded successfully!")
        print()
        
        # Create sample data
        print("📊 Generating sample data...")
        sample_data = create_sample_dataset()
        print(f"✅ Created {len(sample_data)} sample data items")
        print()
        
        # Generate text for each sample
        print("🎯 Generating text from structured data:")
        print("-" * 40)
        
        for i, structured_data in enumerate(sample_data, 1):
            print(f"\n📝 Sample {i}:")
            print(f"   Input Data: {structured_data.data}")
            
            try:
                result = generator.generate_text(structured_data)
                print(f"   Generated Text: {result['generated_text']}")
                
                if i == 1:  # Show device info for first sample
                    print(f"   Device: {result['device']}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print("\n🎉 Demo completed successfully!")
        print("\n💡 Next steps:")
        print("   • Run 'streamlit run web_app/app.py' for the web interface")
        print("   • Run 'python cli.py generate' for command-line usage")
        print("   • Check the README.md for more examples")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("\n🔧 Troubleshooting:")
        print("   • Make sure all dependencies are installed: pip install -r requirements.txt")
        print("   • Check your internet connection for model download")
        print("   • Try a smaller model like 'google/flan-t5-small' if you have memory issues")


if __name__ == "__main__":
    main()
