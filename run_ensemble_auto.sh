#!/bin/bash

# Auto-Restart Ensemble Runner Script
# This script will automatically restart ensemble_95_plus.py if it fails or gets stuck

echo "🎯 Starting Auto-Ensemble Runner..."
echo "🔄 Will automatically restart on failure or if stuck"
echo "📝 Logs will be saved to auto_ensemble_runner.log"
echo ""

# Make sure we're in the right directory
cd /home/mohanganesh/AI_TraceFinder

# Activate virtual environment
source venv/bin/activate

# Run the auto-restart system
python auto_ensemble_runner.py

# Check the result
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Ensemble training completed successfully!"
    echo "📊 Check ensemble_95_results/ for results"
else
    echo ""
    echo "❌ Ensemble training failed after maximum restarts"
    echo "📝 Check auto_ensemble_runner.log for details"
fi
