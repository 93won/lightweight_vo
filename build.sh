#!/bin/bash

# Create build directory
if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# Configure and build
cmake ..
make -j$(nproc)

echo "Build completed!"
echo "Usage: ./test_euroc <path_to_euroc_sequence>"
echo "Example: ./test_euroc /path/to/MH_01_easy"
