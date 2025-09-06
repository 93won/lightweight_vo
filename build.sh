#!/bin/bash

set -e  # Exit on any error

echo "=============================================="
echo "  Lightweight VIO Build Script with ThirdParty"
echo "=============================================="

# Get number of CPU cores for parallel compilation (use half of available cores)
NPROC=$(($(nproc) / 2))
if [ $NPROC -lt 1 ]; then
    NPROC=1
fi
echo "Using $NPROC cores for compilation (half of available)"

# Install system dependencies
echo ""
echo "Step 0: Installing system dependencies..."
echo "========================================"
sudo apt update
sudo apt install -y \
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
    libsuitesparse-dev

echo "System dependencies installed successfully!"

# Build third-party dependencies
echo ""
echo "Step 1: Building third-party dependencies..."
echo "=============================================="

# Build Ceres Solver
echo "Building Ceres Solver..."
if [ ! -d "thirdparty/ceres-solver/build" ]; then
    mkdir -p thirdparty/ceres-solver/build
fi
cd thirdparty/ceres-solver/build
cmake .. \
    -DBUILD_TESTING=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_BENCHMARKS=OFF \
    -DUSE_TBB=OFF \
    -DUSE_OPENMP=ON \
    -DSUITESPARSE=OFF \
    -DCXSPARSE=OFF \
    -DMINIGLOG=ON \
    -DEIGENSPARSE=ON
make -j$NPROC
cd ../../..

# Build Pangolin
echo "Building Pangolin..."
if [ ! -d "thirdparty/pangolin/build" ]; then
    mkdir -p thirdparty/pangolin/build
fi
cd thirdparty/pangolin/build
cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF
make -j$NPROC
cd ../../..

# Build main project
echo ""
echo "Step 2: Building main project..."
echo "================================="

# Create build directory for main project
if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# Configure and build main project
cmake ..
make -j$NPROC

echo ""
echo "=============================================="
echo "  Build completed successfully!"
echo "=============================================="
echo "Executable: ./build/test_vio_viewer"
echo "Usage: ./build/test_vio_viewer <euroc_dataset_path>"
