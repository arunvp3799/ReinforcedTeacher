# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY evaluator.py .
COPY run_evaluation.sh .
RUN chmod +x run_evaluation.sh

# Create results directory
RUN mkdir -p /app/results

# Expose Ollama port
EXPOSE 11434

# Set environment variables
ENV OLLAMA_HOST=0.0.0.0:11434

CMD ["/bin/bash", "run_evaluation.sh"]