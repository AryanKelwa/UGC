# Use standard CUDA base image for PyTorch training
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Setup basic OS utilities
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI v2 (for S3 model uploads)
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf aws awscliv2.zip  

WORKDIR /workspace

# Install requirements
COPY requirements.txt .
COPY requirements/ requirements/
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir -r requirements/training.txt
RUN pip3 install --no-cache-dir -r requirements/inference.txt

# Install Unsloth (Linux + CUDA accelerated LoRA training)
RUN pip3 install --no-cache-dir "unsloth[colab-new]"
RUN pip3 install --no-cache-dir --no-deps xformers trl peft accelerate bitsandbytes triton

# Download NLP model
RUN python3 -m spacy download en_core_web_sm

# Copy codebase
COPY . .

# Set default command to show helper options
CMD ["python3", "main.py", "--help"]
