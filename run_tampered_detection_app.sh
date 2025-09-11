#!/bin/bash

# Tampered Image Detection Streamlit App Launcher
# This script sets up and runs the Streamlit application for tampered image detection

echo "🔍 Starting Tampered Image Detection App..."
echo "=============================================="

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
    "streamlit_tampered_detection_app.py"
    "objective2_feature_extraction.py"
    "objective2_final_optimization_results.csv"
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

# Check if model files exist
echo "🤖 Checking model files..."

model_files=(
    "objective2_baseline_models/scaler.joblib"
    "objective2_baseline_models/feature_info.json"
)

model_missing=()
for file in "${model_files[@]}"; do
    if [ ! -f "$file" ]; then
        model_missing+=("$file")
    fi
done

if [ ${#model_missing[@]} -ne 0 ]; then
    echo "⚠️  Some model files are missing:"
    for file in "${model_missing[@]}"; do
        echo "   - $file"
    done
    echo "The app will use fallback methods."
fi

# Install additional requirements if needed
echo "📥 Installing additional requirements..."
pip install -q streamlit plotly

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p objective2_final_optimization_models
mkdir -p temp_uploads

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Display app information
echo ""
echo "🚀 Launching Tampered Image Detection App..."
echo "=============================================="
echo "📊 Model: LightGBM with SMOTE+ENN"
echo "🎯 Accuracy: 75%"
echo "🔧 Features: 36 forensic features"
echo "📱 Interface: Streamlit Web App"
echo ""
echo "🌐 The app will open in your default browser"
echo "🔗 Local URL: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo "=============================================="

# Run the Streamlit app
streamlit run streamlit_tampered_detection_app.py \
    --server.port 8501 \
    --server.address localhost \
    --browser.gatherUsageStats false \
    --theme.base light \
    --theme.primaryColor "#FF6B6B" \
    --theme.backgroundColor "#FFFFFF" \
    --theme.secondaryBackgroundColor "#F0F2F6" \
    --theme.textColor "#262730"
