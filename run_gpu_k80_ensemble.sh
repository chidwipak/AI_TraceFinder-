#!/bin/bash

echo "🚀 LAUNCHING GPU-OPTIMIZED ENSEMBLE PIPELINE (Tesla K80 Compatible)"
echo "=================================================================="

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check GPU availability
echo "🔍 Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap --format=csv,noheader,nounits

# Check CUDA version
echo "🔍 Checking CUDA version..."
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"
python -c "import cupy as cp; print(f'CuPy CUDA: {cp.cuda.runtime.driverGetVersion()}')"

# Check if K80-compatible libraries are working
echo "🔍 Testing K80-compatible libraries..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import cupy as cp; print(f'CuPy: {cp.__version__}')"

# Run K80-compatible GPU ensemble
echo "🎯 Starting K80-compatible GPU ensemble pipeline..."
python ensemble_95_plus_gpu_k80.py

echo "✅ K80-compatible GPU ensemble pipeline completed!"
