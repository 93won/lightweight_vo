# Installation Guide

This guide provides detailed instructions for installing and building the Lightweight Stereo VIO system.

## Installation Methods

We provide two installation methods:
1. **Native Build (Ubuntu 22.04)** - Direct installation on your system
2. **Docker** - Containerized environment (Recommended)

---

## Method 1: Native Build (Ubuntu 22.04)

This method builds the project and all its dependencies directly on your system. It has been tested on Ubuntu 22.04.

### Prerequisites

- **Ubuntu 22.04 LTS**
- **C++17 Compiler** (g++)
- **CMake** (>= 3.10)
- **Git**

### Step 1: Clone the Repository

```bash
git clone https://github.com/93won/lightweight_vo.git
cd lightweight_vo
```

### Step 2: Run the Build Script

The provided `build.sh` script will automatically install system dependencies (like OpenCV, Eigen, etc.) and then build the third-party libraries (Ceres, Pangolin) and the main application.

```bash
chmod +x build.sh
./build.sh
```

### Step 3: Verify Installation

After successful build, verify that the executable was created:

```bash
ls -la build/
```

You should see the `euroc_stereo` executable in the build directory.

### What the Build Script Does

The `build.sh` script performs the following operations:

1. **System Dependencies Installation**:
   - OpenCV (computer vision library)
   - Eigen3 (linear algebra library)
   - Build essentials (cmake, make, g++)

2. **Third-party Libraries**:
   - **Ceres Solver**: Nonlinear optimization library
   - **Pangolin**: 3D visualization library
   - **Sophus**: Lie group library for robotics
   - **spdlog**: Fast C++ logging library

3. **Project Build**:
   - Creates build directory
   - Runs CMake configuration
   - Compiles the main application

### Troubleshooting Native Build

#### Common Issues

**CMake version too old:**
```bash
# Update CMake if needed
sudo apt remove cmake
sudo snap install cmake --classic
```

**Missing dependencies:**
```bash
# Install missing packages manually
sudo apt update
sudo apt install build-essential cmake git
sudo apt install libeigen3-dev libopencv-dev
```

**Build failures:**
- Check that you have sufficient disk space (at least 5GB)
- Ensure you have internet connection for downloading dependencies
- Try running `build.sh` again if it fails partway through

---

## Method 2: Docker Installation (Recommended)

Using Docker is the recommended method as it provides a self-contained, consistent environment across different systems.

### Prerequisites

- **Docker** installed on your system
- **X11 forwarding** support (for visualization)

### Step 1: Install Docker

If Docker is not installed on your system:

#### Ubuntu/Debian:
```bash
# Install Docker
sudo apt update
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to docker group (optional, avoids sudo)
sudo usermod -aG docker $USER
# Log out and log back in for group changes to take effect
```

#### Other Linux distributions:
Follow the official Docker installation guide for your distribution.

### Step 2: Clone the Repository

```bash
git clone https://github.com/93won/lightweight_vo.git
cd lightweight_vo
```

### Step 3: Build the Docker Image

```bash
chmod +x docker.sh
./docker.sh build
```

This command builds a Docker image named `lightweight-vio:latest` with all necessary dependencies and source code.

### Step 4: Verify Docker Build

Check that the image was created successfully:

```bash
docker images | grep lightweight-vio
```

### What the Docker Build Does

The Docker build process:

1. **Base Image**: Uses Ubuntu 22.04 as base
2. **System Setup**: Installs all required system packages
3. **Dependencies**: Builds all third-party libraries
4. **Application**: Compiles the VIO application
5. **Environment**: Sets up proper environment for X11 forwarding

### Docker Build Troubleshooting

#### Common Issues

**Permission denied:**
```bash
# Add user to docker group or use sudo
sudo ./docker.sh build
```

**Build fails due to network:**
- Check internet connection
- Try building again (Docker caches layers)

**X11 forwarding issues:**
```bash
# Allow X11 connections
xhost +local:docker
```

**Insufficient disk space:**
- Docker images can be large (~2-3GB)
- Clean up unused images: `docker system prune`

---

## System Requirements

### Minimum Requirements
- **CPU**: 4-core processor (Intel i5 or equivalent)
- **RAM**: 8GB system memory
- **Storage**: 10GB available disk space
- **GPU**: Not required (CPU-only implementation)

### Recommended Requirements
- **CPU**: 8-core processor (Intel i7 or equivalent)
- **RAM**: 16GB system memory
- **Storage**: 20GB available disk space
- **GPU**: NVIDIA GPU (for future CUDA acceleration)

## Next Steps

After successful installation:

1. **Download Dataset**: Follow the [Dataset Download Guide](Download_Dataset.md)
2. **Run Examples**: See [Running Examples](Running_Example.md) for usage instructions
3. **Configuration**: Modify config files in `config/` directory as needed

## Additional Notes

### Development Setup

If you plan to modify the code:

```bash
# Install additional development tools
sudo apt install gdb valgrind clang-format
```

### IDE Integration

The project includes CMake configuration files that work well with:
- **VS Code** with C++ extensions
- **CLion** (JetBrains)
- **Qt Creator**

### Performance Optimization

For optimal performance:
- Use Release build configuration
- Enable compiler optimizations
- Consider CPU affinity settings for real-time performance
