# Use NVIDIA CUDA base image with Ubuntu
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Update package lists and install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    htop \
    gnupg \
    lsb-release \
    vim \
    htop \
    tree \
    unzip \
    zip \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip and install common Python packages
RUN pip3 install --upgrade pip setuptools wheel

# Install common scientific Python packages
RUN pip3 install \
    numpy \
    scipy \
    matplotlib \
    pandas \
    jupyter \
    jupyterlab \
    ipython \
    pytest \
    black \
    flake8 \
    mypy

# Install Mojo SDK
# Note: This installs the Mojo SDK via pip (requires authentication with Modular)
RUN pip3 install modular

# Create a workspace directory
WORKDIR /workspace

# Create a non-root user for development
RUN useradd -m -s /bin/bash mojo_dev && \
    usermod -aG sudo mojo_dev && \
    echo "mojo_dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set up the user environment
USER mojo_dev
WORKDIR /home/mojo_dev

# Create common directories
RUN mkdir -p /home/mojo_dev/projects /home/mojo_dev/.local/bin

# Add local bin to PATH
ENV PATH="/home/mojo_dev/.local/bin:$PATH"

# Set up shell environment
RUN echo 'export PATH="/home/mojo_dev/.local/bin:$PATH"' >> /home/mojo_dev/.bashrc

# Switch back to root for final setup
USER root


# Set working directory back to workspace
WORKDIR /workspace

# Change ownership of workspace to mojo_dev user
RUN chown -R mojo_dev:mojo_dev /workspace

# Default command
CMD ["/bin/bash"]