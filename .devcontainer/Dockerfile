# Use NVIDIA CUDA base image with Python 3.11 for GPU support
ARG PYTHON_VERSION=3.11
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install system dependencies and Python
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    sudo \
    wget \
    python3 \
    python3-venv \
    python3-distutils \
    python3-pip \
    ca-certificates \
    coreutils \
    util-linux \
    # Install Node.js LTS for VS Code Remote support
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Create vscode user with sudo privileges
RUN groupadd --gid 1000 vscode \
    && useradd --uid 1000 --gid vscode --shell /bin/bash --create-home vscode \
    && echo 'vscode ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Install uv - the fast Python package installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set environment variables for uv
ENV UV_CACHE_DIR=/opt/uv-cache
ENV UV_PYTHON_PREFERENCE=only-managed
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create cache directory and virtual environment directory
RUN mkdir -p /opt/uv-cache /opt/venv && chown -R vscode:vscode /opt/uv-cache /opt/venv

# Install NVIDIA Container Toolkit requirements (runtime support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Switch to vscode user
USER vscode
WORKDIR /workspace

# Create virtual environment
RUN python3 -m venv /opt/venv

# Upgrade pip and install torch with CUDA support
RUN /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# # Install dependencies using uv with the lock file
# RUN uv sync --frozen

# # Copy the rest of the project
# COPY --chown=vscode:vscode . .

# # Install the project in development mode
# RUN uv pip install -e .

# # Ensure the virtual environment is activated for all sessions
# RUN echo 'source /opt/venv/bin/activate' >> ~/.bashrc
# RUN echo 'source /opt/venv/bin/activate' >> ~/.profile

# # Set default command
# CMD ["/bin/bash"]
