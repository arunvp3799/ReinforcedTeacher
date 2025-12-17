#!/bin/bash

echo "================================================"
echo "LLM Code Generation Evaluation System"
echo "================================================"
echo ""

# Start Ollama service in background
echo "Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
sleep 5

# Check if Ollama is running
max_attempts=30
attempt=0
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ $attempt -eq $max_attempts ]; then
        echo "Error: Ollama failed to start after $max_attempts attempts"
        exit 1
    fi
    echo "Waiting for Ollama... (attempt $attempt/$max_attempts)"
    sleep 2
done

echo "✓ Ollama is running"
echo ""

# Pull required models
echo "Pulling Qwen2.5 3B model..."
ollama pull qwen2.5:3b

echo ""
echo "Pulling Qwen2.5 Coder 3B model..."
ollama pull qwen2.5-coder:3b

echo ""
echo "✓ Models downloaded successfully"
echo ""

# Run the evaluation
echo "Starting evaluation..."
echo "================================================"
python evaluator.py

# Keep container running if evaluation completes
echo ""
echo "Evaluation complete. Check results in /app/results/"
echo "Container will remain running. Press Ctrl+C to stop."

# Wait for Ollama process
wait $OLLAMA_PID