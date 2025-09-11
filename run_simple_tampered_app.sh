#!/bin/bash

# Simple Tampered Image Detection Streamlit App Launcher
# This script runs the simplified Streamlit application

echo "🔍 Starting Simple Tampered Image Detection App..."
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run the setup first."
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if required files exist
echo "🔍 Checking required files..."

required_files=(
    "streamlit_tampered_detection_simple.py"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "❌ Missing required files:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    echo "Please ensure all required files are present."
    exit 1
fi

# Install streamlit if not already installed
echo "📥 Installing Streamlit..."
pip install -q streamlit

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p temp_uploads

# Display app information
echo ""
echo "🚀 Launching Simple Tampered Image Detection App..."
echo "=================================================="
echo "📊 Model: LightGBM with SMOTE+ENN (Simulated)")
echo "🎯 Accuracy: 75%"
echo "🔧 Features: Basic forensic features"
echo "📱 Interface: Streamlit Web App"
echo ""
echo "🌐 The app will open in your default browser"
echo "🔗 Local URL: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo "=================================================="

# Run the Streamlit app
streamlit run streamlit_tampered_detection_simple.py \
    --server.port 8501 \
    --server.address localhost \
    --browser.gatherUsageStats false \
    --theme.base light \
    --theme.primaryColor "#FF6B6B" \
    --theme.backgroundColor "#FFFFFF" \
    --theme.secondaryBackgroundColor "#F0F2F6" \
    --theme.textColor "#262730"
