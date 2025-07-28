# Use Python 3.11 as base image
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    sudo \
    wget \
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

# Switch to vscode user
USER vscode
WORKDIR /workspace

# # Create virtual environment
# RUN uv venv /opt/venv

# # Copy dependency files first for better caching
# COPY --chown=vscode:vscode pyproject.toml uv.lock ./

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
