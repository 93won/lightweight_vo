# Use Ubuntu 20.04 as base image (more stable for builds)
FROM ubuntu:20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopencv-dev \
    libeigen3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglew-dev \
    libyaml-cpp-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /workspace/

# Make build script executable
RUN chmod +x build.sh

# Build the project (skip apt install since dependencies are already installed)
RUN sed -i '/sudo apt update/,/echo "System dependencies installed successfully!"/c\echo "Skipping system dependencies (already installed in Docker image)"' build.sh && \
    ./build.sh

# Expose any ports if needed (adjust as necessary)
# EXPOSE 8080

# Set the entry point - change to build directory first
WORKDIR /workspace/build
ENTRYPOINT ["./euroc_stereo_vo"]

# Default command (can be overridden)
CMD ["--help"]
