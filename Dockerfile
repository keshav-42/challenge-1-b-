FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables for UTF-8 support
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV PYTHONUTF8=1
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model download script
COPY download_models.py .

# Pre-download all models and data during build (requires internet)
# This ensures the container can run offline with --network none
RUN python download_models.py

# Copy the application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p input output output_parsed chunked output_labeled embedded output_vectors && \
    chmod -R 755 input output output_parsed chunked output_labeled embedded output_vectors

# Make the pipeline script executable
RUN chmod +x pipeline.py

# Set environment variable to indicate Docker environment
ENV DOCKER_CONTAINER=1

# Ensure output.json can be written to the output directory
VOLUME ["/app/input", "/app/output"]

# Set the entrypoint to run the pipeline with force-clean and copy output.json to mounted output volume
ENTRYPOINT ["sh", "-c", "python pipeline.py --force-clean \"$@\" && cp -f output.json /app/output/ 2>/dev/null || true", "--"]
