#!/bin/bash

set -e  # Exit on any error

echo "=============================================="
echo "  Lightweight VIO Build Script with ThirdParty"
echo "=============================================="

# Get number of CPU cores for parallel compilation
NPROC=$(nproc)
echo "Using $NPROC cores for compilation"

# Build third-party dependencies
echo ""
echo "Step 1: Building third-party dependencies..."
echo "=============================================="

# Build Ceres Solver
echo "Building Ceres Solver..."
cd thirdparty/ceres-solver
if [ ! -d "build" ]; then
    mkdir build
fi
cd build
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
cd thirdparty/pangolin
if [ ! -d "build" ]; then
    mkdir build
fi
cd build
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
