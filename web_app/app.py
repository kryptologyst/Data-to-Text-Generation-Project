"""
Streamlit web interface for data-to-text generation.
"""

import streamlit as st
import json
import pandas as pd
from typing import Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_to_text_generator import DataToTextGenerator, DataToTextConfig, StructuredData
from config import Config, load_config
from dataset_generator import SyntheticDatasetGenerator, DatasetConfig


def load_generator() -> DataToTextGenerator:
    """Load the data-to-text generator."""
    config = DataToTextConfig(
        model_name="google/flan-t5-base",
        max_length=256,
        temperature=0.7
    )
    return DataToTextGenerator(config)


def create_sample_data() -> Dict[str, List[Dict[str, Any]]]:
    """Create sample data for different categories."""
    return {
        "weather": [
            {
                "city": "New York",
                "temperature": "22°C",
                "humidity": "60%",
                "forecast": "sunny",
                "wind_speed": "10 km/h"
            },
            {
                "city": "London",
                "temperature": "15°C",
                "humidity": "80%",
                "forecast": "cloudy",
                "wind_speed": "15 km/h"
            }
        ],
        "product": [
            {
                "product": "iPhone 15",
                "price": "$999",
                "storage": "128GB",
                "color": "Space Black",
                "availability": "In Stock"
            },
            {
                "product": "MacBook Pro",
                "price": "$1999",
                "storage": "512GB",
                "color": "Silver",
                "availability": "In Stock"
            }
        ],
        "company": [
            {
                "company": "TechCorp",
                "revenue": "$2.5M",
                "employees": "150",
                "industry": "Software",
                "location": "San Francisco"
            },
            {
                "company": "DataSoft",
                "revenue": "$5.2M",
                "employees": "300",
                "industry": "AI/ML",
                "location": "Seattle"
            }
        ],
        "person": [
            {
                "name": "John Smith",
                "age": "32",
                "profession": "Software Engineer",
                "city": "San Francisco",
                "salary": "$120,000"
            },
            {
                "name": "Sarah Johnson",
                "age": "28",
                "profession": "Data Scientist",
                "city": "New York",
                "salary": "$95,000"
            }
        ],
        "event": [
            {
                "event": "AI Conference 2024",
                "date": "2024-06-15",
                "venue": "Convention Center",
                "city": "San Francisco",
                "attendees": "5000"
            },
            {
                "event": "Tech Workshop",
                "date": "2024-07-20",
                "venue": "Hotel Plaza",
                "city": "New York",
                "attendees": "200"
            }
        ]
    }


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Data-to-Text Generation",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("📝 Data-to-Text Generation")
    st.markdown("Convert structured data into natural, human-readable text using AI")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Model selection
    model_options = {
        "FLAN-T5 Base": "google/flan-t5-base",
        "FLAN-T5 Small": "google/flan-t5-small",
        "FLAN-T5 Large": "google/flan-t5-large",
        "T5 Base": "t5-base",
        "T5 Small": "t5-small"
    }
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        index=0
    )
    
    # Generation parameters
    st.sidebar.subheader("Generation Parameters")
    max_length = st.sidebar.slider("Max Length", 50, 512, 256)
    temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
    num_beams = st.sidebar.slider("Number of Beams", 1, 8, 4)
    
    # Load generator
    @st.cache_resource
    def get_generator():
        config = DataToTextConfig(
            model_name=model_options[selected_model],
            max_length=max_length,
            temperature=temperature,
            num_beams=num_beams
        )
        return DataToTextGenerator(config)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Generate Text", "Batch Processing", "Dataset Generator", "Evaluation"])
    
    with tab1:
        st.header("Single Text Generation")
        
        # Category selection
        category = st.selectbox(
            "Select Data Category",
            options=["weather", "product", "company", "person", "event", "custom"]
        )
        
        sample_data = create_sample_data()
        
        if category != "custom":
            # Show sample data
            st.subheader(f"Sample {category.title()} Data")
            sample_items = sample_data.get(category, [])
            
            if sample_items:
                selected_item = st.selectbox(
                    f"Select {category} item",
                    options=range(len(sample_items)),
                    format_func=lambda x: f"Item {x+1}"
                )
                
                data = sample_items[selected_item]
                
                # Display data as JSON
                st.json(data)
                
                # Generate text
                if st.button("Generate Text", type="primary"):
                    with st.spinner("Generating text..."):
                        try:
                            generator = get_generator()
                            structured_data = StructuredData(
                                data=data,
                                context=category
                            )
                            
                            result = generator.generate_text(structured_data)
                            
                            st.success("Text generated successfully!")
                            st.subheader("Generated Text:")
                            st.write(result['generated_text'])
                            
                            # Show metadata
                            with st.expander("Generation Details"):
                                st.json({
                                    "model": result['model'],
                                    "device": result['device'],
                                    "generation_params": result['generation_params']
                                })
                        
                        except Exception as e:
                            st.error(f"Error generating text: {e}")
        
        else:
            # Custom data input
            st.subheader("Custom Data Input")
            custom_data = st.text_area(
                "Enter JSON data",
                value='{"key": "value", "another_key": "another_value"}',
                height=200
            )
            
            try:
                data = json.loads(custom_data)
                st.json(data)
                
                if st.button("Generate Text from Custom Data", type="primary"):
                    with st.spinner("Generating text..."):
                        try:
                            generator = get_generator()
                            structured_data = StructuredData(data=data)
                            
                            result = generator.generate_text(structured_data)
                            
                            st.success("Text generated successfully!")
                            st.subheader("Generated Text:")
                            st.write(result['generated_text'])
                        
                        except Exception as e:
                            st.error(f"Error generating text: {e}")
            
            except json.JSONDecodeError:
                st.error("Invalid JSON format. Please check your input.")
    
    with tab2:
        st.header("Batch Processing")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload JSON file with data",
            type=['json'],
            help="Upload a JSON file containing an array of data objects"
        )
        
        if uploaded_file:
            try:
                data = json.load(uploaded_file)
                
                if isinstance(data, list):
                    st.success(f"Loaded {len(data)} data items")
                    
                    # Show preview
                    st.subheader("Data Preview")
                    df = pd.DataFrame(data)
                    st.dataframe(df.head())
                    
                    # Batch processing
                    if st.button("Process All Items", type="primary"):
                        with st.spinner("Processing batch..."):
                            try:
                                generator = get_generator()
                                
                                structured_data_list = [
                                    StructuredData(data=item) for item in data
                                ]
                                
                                results = generator.batch_generate(structured_data_list)
                                
                                st.success(f"Processed {len(results)} items successfully!")
                                
                                # Display results
                                st.subheader("Results")
                                for i, result in enumerate(results):
                                    with st.expander(f"Item {i+1}"):
                                        st.write("**Generated Text:**")
                                        st.write(result['generated_text'])
                                        st.write("**Input Data:**")
                                        st.json(result['prompt'])
                                
                                # Download results
                                results_json = json.dumps(results, indent=2)
                                st.download_button(
                                    label="Download Results",
                                    data=results_json,
                                    file_name="batch_results.json",
                                    mime="application/json"
                                )
                            
                            except Exception as e:
                                st.error(f"Error processing batch: {e}")
                
                else:
                    st.error("JSON file must contain an array of objects")
            
            except json.JSONDecodeError:
                st.error("Invalid JSON file")
    
    with tab3:
        st.header("Synthetic Dataset Generator")
        
        st.markdown("Generate synthetic datasets for training and evaluation.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_samples = st.number_input("Number of samples", 100, 5000, 1000)
            categories = st.multiselect(
                "Categories",
                options=["weather", "product", "company", "person", "event"],
                default=["weather", "product", "company"]
            )
        
        with col2:
            random_seed = st.number_input("Random seed", 0, 999999, 42)
            output_format = st.selectbox("Output format", ["JSON", "CSV", "Both"])
        
        if st.button("Generate Dataset", type="primary"):
            with st.spinner("Generating dataset..."):
                try:
                    config = DatasetConfig(
                        num_samples=num_samples,
                        categories=categories,
                        random_seed=random_seed
                    )
                    
                    generator = SyntheticDatasetGenerator(config)
                    data = generator.generate_dataset()
                    reference_texts = generator.create_reference_texts(data)
                    
                    st.success(f"Generated {len(data)} samples successfully!")
                    
                    # Show statistics
                    category_counts = {}
                    for item in data:
                        category = item["category"]
                        category_counts[category] = category_counts.get(category, 0) + 1
                    
                    # Create pie chart
                    fig = px.pie(
                        values=list(category_counts.values()),
                        names=list(category_counts.keys()),
                        title="Dataset Distribution by Category"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show sample
                    st.subheader("Sample Data")
                    sample_df = pd.DataFrame(data[:5])
                    st.dataframe(sample_df)
                    
                    # Download options
                    if output_format in ["JSON", "Both"]:
                        json_data = json.dumps(data, indent=2)
                        st.download_button(
                            label="Download JSON",
                            data=json_data,
                            file_name="synthetic_dataset.json",
                            mime="application/json"
                        )
                    
                    if output_format in ["CSV", "Both"]:
                        csv_data = pd.DataFrame(data).to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv_data,
                            file_name="synthetic_dataset.csv",
                            mime="text/csv"
                        )
                
                except Exception as e:
                    st.error(f"Error generating dataset: {e}")
    
    with tab4:
        st.header("Model Evaluation")
        
        st.markdown("Evaluate model performance using ROUGE metrics.")
        
        # Upload generated and reference texts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Generated Texts")
            generated_texts = st.text_area(
                "Enter generated texts (one per line)",
                height=200,
                help="Enter one generated text per line"
            )
        
        with col2:
            st.subheader("Reference Texts")
            reference_texts = st.text_area(
                "Enter reference texts (one per line)",
                height=200,
                help="Enter one reference text per line"
            )
        
        if st.button("Evaluate", type="primary"):
            if generated_texts and reference_texts:
                try:
                    gen_list = [text.strip() for text in generated_texts.split('\n') if text.strip()]
                    ref_list = [text.strip() for text in reference_texts.split('\n') if text.strip()]
                    
                    if len(gen_list) != len(ref_list):
                        st.error("Number of generated and reference texts must match")
                    else:
                        with st.spinner("Evaluating..."):
                            generator = get_generator()
                            metrics = generator.evaluate_generation(gen_list, ref_list)
                            
                            if metrics:
                                st.success("Evaluation completed!")
                                
                                # Display metrics
                                st.subheader("ROUGE Scores")
                                
                                metric_data = {
                                    "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L", "ROUGE-Lsum"],
                                    "Score": [
                                        f"{metrics['rouge1']:.4f}",
                                        f"{metrics['rouge2']:.4f}",
                                        f"{metrics['rougeL']:.4f}",
                                        f"{metrics['rougeLsum']:.4f}"
                                    ]
                                }
                                
                                df = pd.DataFrame(metric_data)
                                st.dataframe(df, use_container_width=True)
                                
                                # Create bar chart
                                fig = go.Figure(data=[
                                    go.Bar(
                                        x=list(metrics.keys()),
                                        y=list(metrics.values()),
                                        text=[f"{v:.4f}" for v in metrics.values()],
                                        textposition='auto'
                                    )
                                ])
                                
                                fig.update_layout(
                                    title="ROUGE Scores",
                                    xaxis_title="Metric",
                                    yaxis_title="Score",
                                    yaxis=dict(range=[0, 1])
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            else:
                                st.error("Failed to compute evaluation metrics")
                
                except Exception as e:
                    st.error(f"Error during evaluation: {e}")
            else:
                st.warning("Please enter both generated and reference texts")


if __name__ == "__main__":
    main()
