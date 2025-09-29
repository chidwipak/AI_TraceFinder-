#!/bin/bash

# AI TraceFinder - Combined Streamlit Application Launch Script
# Launches the combined scanner identification and tampered detection app

echo "🔍 AI TraceFinder - Combined Application"
echo "========================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install/update requirements
echo "📦 Installing/updating requirements..."
pip install -r requirements_combined.txt

# Check if required model files exist
echo "🔍 Checking model files..."

# Scanner model files
scanner_model_files=(
    "ensemble_95_results_ultra/models/Stacking_Top5_Ridge_model.pkl"
    "ensemble_95_results_ultra/imputer.pkl"
    "ensemble_95_results_ultra/feature_selector.pkl"
    "ensemble_95_results_ultra/label_encoder.pkl"
)

# Tampered detection model files
tampered_model_files=(
    "new_objective2_ensemble_results/models/best_ensemble_meta_learner.pkl"
    "new_objective2_baseline_ml_results/models/best_model.pkl"
    "new_objective2_baseline_ml_results/models/scaler.pkl"
)

# Check scanner model files
missing_files=()
for file in "${scanner_model_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

# Check tampered model files
for file in "${tampered_model_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    echo "⚠️  Warning: Some model files are missing:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    echo ""
    echo "Please ensure all model files are present before running the application."
    echo "The app will show errors if models cannot be loaded."
    echo ""
fi

# Check if PRNU fingerprints exist
if [ ! -f "prnu_fingerprints/fingerprints/all_prnu_fingerprints.pkl" ]; then
    echo "⚠️  Warning: PRNU fingerprints not found."
    echo "   The scanner identification will use fallback features."
    echo ""
fi

# Create necessary directories if they don't exist
mkdir -p logs
mkdir -p temp

# Set environment variables
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Launch Streamlit app
echo "🚀 Launching AI TraceFinder Combined Application..."
echo "📍 URL: http://localhost:8501"
echo "🔄 Press Ctrl+C to stop the application"
echo ""

streamlit run streamlit_combined_app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=false \
    --browser.gatherUsageStats=false \
    --theme.base="light" \
    --theme.primaryColor="#1f77b4" \
    --theme.backgroundColor="#ffffff" \
    --theme.secondaryBackgroundColor="#f0f2f6"
