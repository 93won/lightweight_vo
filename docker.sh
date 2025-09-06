#!/bin/bash

# Docker build and run script for lightweight VO

IMAGE_NAME="lightweight-vo"
TAG="latest"

echo "=============================================="
echo "  Lightweight VO Docker Build Script"
echo "=============================================="

# Function to build Docker image
build_image() {
    echo "Building Docker image: $IMAGE_NAME:$TAG"
    docker build -t $IMAGE_NAME:$TAG .
    
    if [ $? -eq 0 ]; then
        echo "Docker image built successfully!"
        echo "Image name: $IMAGE_NAME:$TAG"
    else
        echo "Failed to build Docker image"
        exit 1
    fi
}

# Function to run Docker container
run_container() {
    local dataset_path=$1
    
    if [ -z "$dataset_path" ]; then
        echo "Usage: $0 run <euroc_dataset_path>"
        echo "Example: $0 run /path/to/euroc/MH_01_easy"
        exit 1
    fi
    
    if [ ! -d "$dataset_path" ]; then
        echo "Error: Dataset path does not exist: $dataset_path"
        exit 1
    fi
    
    echo "Running Docker container with dataset: $dataset_path"
    docker run --rm -it \
        -v "$dataset_path:/dataset:ro" \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        --privileged \
        $IMAGE_NAME:$TAG /dataset
}

# Function to run interactive shell in container
run_shell() {
    echo "Starting interactive shell in Docker container"
    docker run --rm -it \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        --privileged \
        --entrypoint /bin/bash \
        $IMAGE_NAME:$TAG
}

# Main script logic
case "$1" in
    "build")
        build_image
        ;;
    "run")
        run_container "$2"
        ;;
    "shell")
        run_shell
        ;;
    "clean")
        echo "Removing Docker image: $IMAGE_NAME:$TAG"
        docker rmi $IMAGE_NAME:$TAG
        ;;
    *)
        echo "Usage: $0 {build|run|shell|clean}"
        echo ""
        echo "Commands:"
        echo "  build           Build the Docker image"
        echo "  run <dataset>   Run VIO with dataset (mount dataset directory)"
        echo "  shell           Start interactive shell in container"
        echo "  clean           Remove the Docker image"
        echo ""
        echo "Examples:"
        echo "  $0 build"
        echo "  $0 run /path/to/euroc/MH_01_easy"
        echo "  $0 shell"
        echo "  $0 clean"
        exit 1
        ;;
esac
