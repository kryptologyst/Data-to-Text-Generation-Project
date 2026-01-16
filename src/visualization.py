"""
Visualization utilities for data-to-text generation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_generation_metrics(metrics: Dict[str, float], title: str = "Generation Metrics") -> go.Figure:
    """
    Create a bar chart of generation metrics.
    
    Args:
        metrics: Dictionary of metric names and values
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure(data=[
        go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            text=[f"{v:.4f}" for v in metrics.values()],
            textposition='auto',
            marker_color='lightblue'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Metric",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        template="plotly_white"
    )
    
    return fig


def plot_dataset_distribution(data: List[Dict[str, Any]], title: str = "Dataset Distribution") -> go.Figure:
    """
    Create a pie chart showing dataset distribution by category.
    
    Args:
        data: List of data items with 'category' field
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    category_counts = {}
    for item in data:
        category = item.get("category", "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
    
    fig = px.pie(
        values=list(category_counts.values()),
        names=list(category_counts.keys()),
        title=title,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    return fig


def plot_text_length_distribution(texts: List[str], title: str = "Text Length Distribution") -> go.Figure:
    """
    Create a histogram of text lengths.
    
    Args:
        texts: List of text strings
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    lengths = [len(text.split()) for text in texts]
    
    fig = go.Figure(data=[
        go.Histogram(
            x=lengths,
            nbinsx=20,
            marker_color='lightgreen',
            opacity=0.7
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Word Count",
        yaxis_title="Frequency",
        template="plotly_white"
    )
    
    return fig


def plot_generation_comparison(
    generated_texts: List[str], 
    reference_texts: List[str],
    title: str = "Generated vs Reference Text Lengths"
) -> go.Figure:
    """
    Compare generated and reference text lengths.
    
    Args:
        generated_texts: List of generated texts
        reference_texts: List of reference texts
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    gen_lengths = [len(text.split()) for text in generated_texts]
    ref_lengths = [len(text.split()) for text in reference_texts]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=ref_lengths,
        y=gen_lengths,
        mode='markers',
        name='Generated vs Reference',
        marker=dict(
            color='blue',
            size=8,
            opacity=0.6
        )
    ))
    
    # Add diagonal line for perfect correlation
    max_length = max(max(gen_lengths), max(ref_lengths))
    fig.add_trace(go.Scatter(
        x=[0, max_length],
        y=[0, max_length],
        mode='lines',
        name='Perfect Correlation',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Reference Text Length (words)",
        yaxis_title="Generated Text Length (words)",
        template="plotly_white"
    )
    
    return fig


def create_evaluation_dashboard(
    metrics: Dict[str, float],
    generated_texts: List[str],
    reference_texts: List[str],
    data: List[Dict[str, Any]]
) -> go.Figure:
    """
    Create a comprehensive evaluation dashboard.
    
    Args:
        metrics: Evaluation metrics
        generated_texts: List of generated texts
        reference_texts: List of reference texts
        data: Original structured data
        
    Returns:
        Plotly figure with subplots
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("ROUGE Metrics", "Dataset Distribution", 
                       "Text Length Distribution", "Length Comparison"),
        specs=[[{"type": "bar"}, {"type": "pie"}],
               [{"type": "histogram"}, {"type": "scatter"}]]
    )
    
    # ROUGE metrics
    fig.add_trace(
        go.Bar(x=list(metrics.keys()), y=list(metrics.values()), 
               name="ROUGE Scores", marker_color='lightblue'),
        row=1, col=1
    )
    
    # Dataset distribution
    category_counts = {}
    for item in data:
        category = item.get("category", "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
    
    fig.add_trace(
        go.Pie(labels=list(category_counts.keys()), 
               values=list(category_counts.values()),
               name="Categories"),
        row=1, col=2
    )
    
    # Text length distribution
    gen_lengths = [len(text.split()) for text in generated_texts]
    fig.add_trace(
        go.Histogram(x=gen_lengths, name="Generated Lengths", 
                    marker_color='lightgreen'),
        row=2, col=1
    )
    
    # Length comparison
    ref_lengths = [len(text.split()) for text in reference_texts]
    fig.add_trace(
        go.Scatter(x=ref_lengths, y=gen_lengths, mode='markers',
                  name="Length Comparison", marker_color='purple'),
        row=2, col=2
    )
    
    fig.update_layout(
        title="Data-to-Text Generation Evaluation Dashboard",
        height=800,
        showlegend=False
    )
    
    return fig


def save_plots(figures: Dict[str, go.Figure], output_dir: str = "plots") -> None:
    """
    Save multiple plots to files.
    
    Args:
        figures: Dictionary of plot names and figure objects
        output_dir: Output directory for plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for name, fig in figures.items():
        filename = f"{output_dir}/{name}.html"
        fig.write_html(filename)
        print(f"Saved plot: {filename}")


def main():
    """Demo the visualization functions."""
    # Sample data for demonstration
    sample_metrics = {
        "rouge1": 0.45,
        "rouge2": 0.32,
        "rougeL": 0.42,
        "rougeLsum": 0.44
    }
    
    sample_data = [
        {"category": "weather", "city": "New York"},
        {"category": "weather", "city": "London"},
        {"category": "product", "name": "iPhone"},
        {"category": "product", "name": "Samsung"},
        {"category": "company", "name": "TechCorp"}
    ]
    
    sample_generated = [
        "The weather in New York is sunny with 22°C temperature.",
        "London has cloudy weather with 15°C temperature.",
        "The iPhone is a premium smartphone with advanced features.",
        "Samsung offers innovative technology and design.",
        "TechCorp is a leading technology company."
    ]
    
    sample_reference = [
        "New York weather: sunny, 22°C",
        "London weather: cloudy, 15°C", 
        "iPhone: premium smartphone",
        "Samsung: innovative technology",
        "TechCorp: leading tech company"
    ]
    
    # Create plots
    plots = {
        "metrics": plot_generation_metrics(sample_metrics),
        "distribution": plot_dataset_distribution(sample_data),
        "length_dist": plot_text_length_distribution(sample_generated),
        "comparison": plot_generation_comparison(sample_generated, sample_reference),
        "dashboard": create_evaluation_dashboard(sample_metrics, sample_generated, sample_reference, sample_data)
    }
    
    # Save plots
    save_plots(plots)
    print("Visualization demo completed!")


if __name__ == "__main__":
    main()
