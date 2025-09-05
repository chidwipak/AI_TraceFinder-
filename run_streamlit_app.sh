#!/bin/bash
# Run the AI TraceFinder Streamlit App

echo "🚀 Starting AI TraceFinder - Scanner Identification App"
echo "=================================================="

# Activate virtual environment
source venv/bin/activate

# Check if model file exists
if [ ! -f "ensemble_95_results_ultra/models/Stacking_Top5_Ridge_model.pkl" ]; then
    echo "❌ Error: Model file not found!"
    echo "Please ensure the Stacking_Top5_Ridge_model.pkl file exists in ensemble_95_results_ultra/models/"
    exit 1
fi

echo "✅ Model file found"
echo "📱 Starting Streamlit app..."
echo ""
echo "🌐 The app will open in your browser at: http://localhost:8501"
echo "📋 Features:"
echo "   - Upload TIFF, PDF, JPG, PNG files"
echo "   - Identify 11 supported scanner models"
echo "   - Detect unknown scanners with confidence threshold"
echo "   - 93.75% accuracy with Stacking_Top5_Ridge model"
echo ""
echo "Press Ctrl+C to stop the app"
echo ""

# Run Streamlit app
streamlit run streamlit_scanner_app.py --server.port 8501 --server.address 0.0.0.0
