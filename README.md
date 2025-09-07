# Lightweight Stereo VO

This is a lightweight stereo visual odometry (VO) project designed for real-time performance. It utilizes feature tracking, sliding window optimization with Ceres Solver, and Pangolin for visualization.

## Features

- Stereo vision-based visual odometry
- Sliding window optimization using Ceres Solver
- Real-time visualization with Pangolin
- Docker support for easy deployment and testing

## Demo

[![Stereo VO Demo](https://img.youtube.com/vi/fM0tq-6E8fg/0.jpg)](https://youtu.be/fM0tq-6E8fg)

## To-Do List

- [ ] **IMU Integration**: Incorporate IMU data to create a more robust Visual-Inertial Odometry (VIO) system, improving accuracy and handling of fast motions.
- [ ] **Embedded Environment Testing**: Test and optimize performance on embedded platforms like Jetson Nano, Raspberry Pi, and other resource-constrained devices.
- [ ] **Fisheye Camera Support**: Add support for fisheye camera models and distortion correction for wide-angle stereo setups.


## 1. Native Build (Ubuntu 22.04)

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

The provided `build.sh` script will automatically install system dependencies (like OpenCV, Eigen, etc.) and then build the third-party libraries (Ceres, Pangolin) and the main VIO application.

```bash
chmod +x build.sh
./build.sh
```

### Step 3: Prepare the EuRoC Dataset

The repository includes a convenience script to download the `MH_01_easy` sequence.

```bash
chmod +x script/download_euroc.sh
./script/download_euroc.sh /path/to/download/
```
This will download the sequence into the specified directory path. You can modify the script to download other sequences.

### Step 4: Run the VIO

After the build is complete, you can run the VIO with a EuRoC dataset. The configuration file (`config/euroc.yaml`) is loaded automatically by the executable.

```bash
./build/euroc_stereo_vo <path_to_euroc_dataset>
```

**Example:**
```bash
./build/euroc_stereo_vo dataset/MH_01_easy
```

---

## 2. Docker Usage

Using Docker is the recommended method as it provides a self-contained, consistent environment. The `docker.sh` script simplifies the process.

### Step 1: Prepare the EuRoC Dataset

First, download the dataset on your host machine. The repository includes a convenience script to download the `MH_01_easy` sequence.

```bash
chmod +x script/download_euroc.sh
./script/download_euroc.sh /path/to/download/
```
This will download the sequence into the specified directory path. This directory will be mounted into the Docker container.

### Step 2: Build the Docker Image

This command builds a Docker image named `lightweight-vio:latest` with all necessary dependencies and source code.

```bash
./docker.sh build
```

### Step 3: Run the VIO in a Container

This command runs the VIO inside a new container. It mounts your local dataset directory into the container (read-only) and forwards your X11 display for the Pangolin viewer.

```bash
./docker.sh run <path_to_euroc_dataset_on_host>
```

**Example:**
```bash
# Assuming you downloaded the dataset to the default 'dataset' directory
./docker.sh run $(pwd)/dataset/MH_01_easy
```
The container will automatically execute the VIO with the provided dataset.

### Additional Docker Commands

- **Start an interactive shell in the container:**
  This is useful for debugging or running commands manually inside the container.
  ```bash
  ./docker.sh shell
  ```

- **Clean up the Docker image:**
  This will remove the `lightweight-vio:latest` image from your system.
  ```bash
  ./docker.sh clean
  ```

## Project Structure

The source code is organized into the following directories:

- `app/`: Main application entry point (`euroc_stereo_vo.cpp`)
- `src/`:
  - `database/`: Data structures for frames, map points, and features.
  - `processing/`: Core VIO processing modules (feature tracking, optimization).
  - `optimization/`: Ceres Solver cost functions (factors) and parameter blocks.
  - `viewer/`: Pangolin-based visualization.
  - `util/`: Utility functions for configuration and data loading.
- `thirdparty/`: External libraries (Ceres, Pangolin, Sophus, spdlog).
- `config/`: Configuration files.
- `scripts/`: Helper scripts.
