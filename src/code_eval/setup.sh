#!/bin/bash

# Simple setup script for code evaluation

set -e

echo "================================"
echo "Code Evaluation Setup"
echo "================================"
echo

# Step 1: Install dependencies
echo "Step 1: Installing Python dependencies..."
pip install -q numpy tqdm
echo "✓ Dependencies installed"
echo

# Step 2: Build Docker image
echo "Step 2: Building Docker sandbox image..."
docker build -t humaneval-sandbox:latest -f Dockerfile . -q
echo "✓ Docker image built"
echo

# Step 3: Test the setup
echo "Step 3: Testing setup..."
python docker_executor.py > /dev/null 2>&1 && echo "✓ Docker executor test passed" || echo "⚠ Docker executor test failed"
python metrics.py > /dev/null 2>&1 && echo "✓ Metrics test passed" || echo "⚠ Metrics test failed"
echo

echo "================================"
echo "Setup Complete!"
echo "================================"
echo
echo "Ready to evaluate your results!"
echo
echo "Example usage:"
echo "  python evaluate_results.py \\"
echo "      --input ../../results/your_model_humaneval.json \\"
echo "      --n-workers 4"
echo
