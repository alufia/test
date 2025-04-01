# Use NVIDIA CUDA 12.1 as base image with Ubuntu 22.04
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python 3.12
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Update alternatives to make Python 3.12 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Update pip to latest version
RUN python3 -m pip install --upgrade pip

# Install common Python dependencies (optional - modify as needed)
RUN pip install \
    numpy \
    torch \
    torchvision \
    cudatoolkit

# Copy your application files (uncomment and modify as needed)
# COPY requirements.txt .
# RUN pip install -r requirements.txt
# COPY . .

# Set default command
CMD ["python3"]
