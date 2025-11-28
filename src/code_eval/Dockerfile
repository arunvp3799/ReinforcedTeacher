FROM python:3.10-slim

# Install minimal required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for execution
RUN useradd -m -u 1000 sandbox && \
    mkdir -p /sandbox && \
    chown -R sandbox:sandbox /sandbox

# Set working directory
WORKDIR /sandbox

# Switch to non-root user
USER sandbox

# Set resource limits via ulimit (enforced by Docker)
# Memory and CPU limits should be set when running the container

# Entry point will execute the provided Python script
ENTRYPOINT ["python3", "-u"]
