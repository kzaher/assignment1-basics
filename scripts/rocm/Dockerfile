FROM ubuntu:22.04

# Accept ROCM_VERSION as a build argument with default value
ARG ROCM_VERSION=6.4

RUN set -e

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install required packages
RUN apt update
RUN apt install -y ca-certificates wget gnupg

# Add ROCm repository key using the new method (apt-key is deprecated)
RUN wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/trusted.gpg.d/rocm.gpg

# Add ROCm repository with proper priority
RUN echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/${ROCM_VERSION}/ jammy main" | tee /etc/apt/sources.list.d/rocm.list

# Set ROCm repository priority to ensure ROCm packages are preferred
RUN echo 'Package: *' > /etc/apt/preferences.d/rocm-pin-600 && \
    echo 'Pin: release o=repo.radeon.com' >> /etc/apt/preferences.d/rocm-pin-600 && \
    echo 'Pin-Priority: 600' >> /etc/apt/preferences.d/rocm-pin-600

# Remove any conflicting packages that might be installed
RUN apt remove -y rocm-cmake rocm-device-libs || true

# Update package list with new repository
RUN apt update

# Install ROCm core packages first
RUN apt install -y rocm-core rocm-utils rocm-cmake rocm-device-libs

# Install ROCm development packages
RUN apt install -y rocm-dev hip-dev rocblas-dev rocsparse-dev rocrand-dev rocfft-dev

# Add ROCm to PATH
RUN echo 'export PATH=$PATH:/opt/rocm/bin' >> /etc/bash.bashrc

# Verify installation
# RUN /opt/rocm/bin/hipcc --version