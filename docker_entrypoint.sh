#!/bin/bash
set -e

# Default values
APP_TYPE="vio"
DATASET_PATH=""

# Parse arguments
if [ "$1" = "vo" ] || [ "$1" = "vio" ]; then
    APP_TYPE=$1
    shift
fi

if [ -n "$1" ]; then
    DATASET_PATH=$1
fi

# Determine which executable and config to use
if [ "$APP_TYPE" = "vio" ]; then
    EXECUTABLE="./euroc_stereo_vio"
    CONFIG_FILE="/workspace/config/euroc_vio.yaml"
else
    EXECUTABLE="./euroc_stereo_vo"
    CONFIG_FILE="/workspace/config/euroc_vo.yaml"
fi

# Change to the build directory
cd /workspace/build

# Check if a dataset path is provided
if [ -z "$DATASET_PATH" ]; then
    echo "No dataset path provided. Running with --help."
    exec $EXECUTABLE --help
else
    echo "Running $APP_TYPE with dataset: $DATASET_PATH"
    exec $EXECUTABLE $CONFIG_FILE $DATASET_PATH
fi
