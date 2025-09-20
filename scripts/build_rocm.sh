#!/bin/bash
set -e

# ROCm version tag
ROCM_VERSION="6.4"
IMAGE_NAME="cs336_rocm"

pushd scripts/rocm
docker build --build-arg ROCM_VERSION=${ROCM_VERSION} . -t ${IMAGE_NAME}:${ROCM_VERSION}
docker tag ${IMAGE_NAME}:${ROCM_VERSION} kruno/${IMAGE_NAME}:${ROCM_VERSION}
docker push kruno/${IMAGE_NAME}:${ROCM_VERSION}