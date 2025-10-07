#!/usr/bin/env python3
"""
Simple Gradio Application for Tampered Image Detection
Minimal version for testing Hugging Face deployment
"""

import gradio as gr
import numpy as np
from PIL import Image
import random

def analyze_image(image):
    """Simple analysis function that returns random results for testing"""
    if image is None:
        return "Please upload an image", None, None
    
    # Simulate analysis
    # Randomly determine if image is tampered (70% chance of original, 30% chance of tampered)
    is_tampered = random.random() < 0.3
    
    if is_tampered:
        result_text = "🚨 TAMPERED IMAGE DETECTED"
        confidence = random.uniform(0.7, 0.95)
    else:
        result_text = "✅ ORIGINAL IMAGE"
        confidence = random.uniform(0.6, 0.9)
    
    # Create a simple plot
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(6, 4))
    categories = ['Original', 'Tampered']
    values = [1-confidence, confidence] if is_tampered else [confidence, 1-confidence]
    colors = ['green', 'red']
    
    bars = ax.bar(categories, values, color=colors)
    ax.set_ylabel('Confidence')
    ax.set_title('Detection Confidence')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    return result_text, confidence, fig

def create_interface():
    """Create Gradio interface"""
    with gr.Blocks(title="Tampered Image Detection") as interface:
        gr.Markdown("""
        # 🔍 Tampered Image Detection System
        
        Upload an image to detect if it has been tampered with.
        
        **Note:** This is a demo version. In the full version, it would use a trained ML model.
        """)
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")
                analyze_btn = gr.Button("Analyze Image", variant="primary")
            
            with gr.Column():
                result_output = gr.Textbox(label="Result")
                confidence_output = gr.Number(label="Confidence Score")
                plot_output = gr.Plot(label="Confidence Visualization")
        
        analyze_btn.click(
            fn=analyze_image,
            inputs=image_input,
            outputs=[result_output, confidence_output, plot_output]
        )
        
        gr.Markdown("""
        ## How It Works
        
        **In the full version, this system would:**
        - Extract 105 forensic features from the uploaded image
        - Use a trained ensemble model to make predictions
        - Provide confidence scores and detailed analysis
        
        **Current Demo Features:**
        - Simple UI for image upload and analysis
        - Random results for demonstration purposes
        """)
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )