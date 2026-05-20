# Use standard CUDA base image for PyTorch training
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Setup basic OS utilities
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install requirements
COPY requirements.txt .
COPY requirements/ requirements/
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir -r requirements/training.txt
RUN pip3 install --no-cache-dir -r requirements/inference.txt

# Download NLP model
RUN python3 -m spacy download en_core_web_sm

# Copy codebase
COPY . .

# Set default command to show helper options
CMD ["python3", "main.py", "--help"]
