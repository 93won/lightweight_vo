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
- [ ] **Fisheye Camera Support**: Add support for fisheye camera models and distortion correction for wide-angle stereo setups.
- [ ] **Embedded Environment Testing**: Test and optimize performance on embedded platforms like Jetson Nano, NUC, and other resource-constrained devices.

## Installation

### 1. Native Build (Ubuntu 22.04)

This method builds the project and all its dependencies directly on your system. It has been tested on Ubuntu 22.04.

### Prerequisites

- **Ubuntu 22.04 LTS**
- **C++17 Compiler** (g++)
- **CMake** (>= 3.10)
- **Git**

### Step 1-1: Clone the Repository

```bash
git clone https://github.com/93won/lightweight_vo.git
cd lightweight_vo
```

### Step 1-2: Run the Build Script

The provided `build.sh` script will automatically install system dependencies (like OpenCV, Eigen, etc.) and then build the third-party libraries (Ceres, Pangolin) and the main VIO application.

```bash
chmod +x build.sh
./build.sh
```

### Step 1-3: Prepare the EuRoC Dataset

The repository includes a convenience script to download all EuRoC MAV datasets (11 sequences total: MH_01-05 and V1_01-03, V2_01-03).

```bash
chmod +x script/download_euroc.sh
./script/download_euroc.sh /path/of/dataset
```
This will download all sequences into a `/path/of/dataset` directory at the root of the project. The script automatically skips sequences that are already downloaded.

### Step 1-4: Run the VO

After the build is complete, you can run the VO with a EuRoC dataset. You need to provide both the configuration file path and the dataset path as arguments.

```bash
./build/euroc_stereo_vo <config_file_path> <euroc_dataset_path>
```

**Example:**
```bash
./build/euroc_stereo_vo /home/lightweight_vo/config/euroc.yaml /home/dataset/EuRoC/MH_01_easy
```

---

### 2. Docker Usage

Using Docker is the recommended method as it provides a self-contained, consistent environment. The `docker.sh` script simplifies the process.

### Step 2-1: Prepare the EuRoC Dataset

First, download the dataset on your host machine. The repository includes a convenience script to download all EuRoC MAV datasets (11 sequences total).

```bash
chmod +x script/download_euroc.sh
./script/download_euroc.sh /path/of/dataset
```
This will download all sequences into a `/path/of/dataset` directory at the root of the project. This directory will be mounted into the Docker container.

### Step 2-2: Build the Docker Image

This command builds a Docker image named `lightweight-vio:latest` with all necessary dependencies and source code.

```bash
./docker.sh build
```

### Step 2-3: Run the VIO in a Container

This command runs the VIO inside a new container. It mounts your local dataset directory into the container (read-only) and forwards your X11 display for the Pangolin viewer. The container automatically uses the built-in configuration file and the provided dataset path.

```bash
./docker.sh run <path_to_euroc_dataset_on_host>
```

**Example:**
```bash
# Assuming you downloaded the dataset to the default 'dataset' directory
./docker.sh run /home/dataset/EuRoC/MH_01_easy
```
The container will automatically execute the VIO with the built-in config file and the provided dataset.



## Performance Analysis and Evaluation

The application automatically performs comprehensive analysis of the visual odometry results and outputs detailed statistics upon completion.

### 1. Built-in Analysis Features

**1-1. Frame-to-Frame Transform Error Analysis**
- Calculates rotation and translation errors between consecutive frames when ground truth is available
- Provides statistical metrics: mean, median, min, max, and RMSE
- Outputs beautifully formatted results with rotation errors in degrees and translation errors in meters

**1-2. Timing Analysis** 
- Measures frame processing times throughout the entire sequence
- Reports average processing time in milliseconds and equivalent FPS
- Helps evaluate real-time performance capabilities

**1-3. Trajectory Output**
- Saves both ground truth and estimated trajectories in TUM format
- Files are automatically generated for further analysis with external tools

### 2. External Evaluation with EVO

For more comprehensive trajectory evaluation, you can use the [EVO (Python package for the evaluation of odometry and SLAM)](https://github.com/MichaelGrupp/evo) tool with the generated trajectory files.

**Installation:**
```bash
pip install evo
```

**Example Usage:**
```bash
# Compare estimated trajectory against ground truth using Relative Pose Error (RPE)
evo_rpe tum ground_truth.txt estimated_trajectory.txt --plot --save_results results/

# Compare using Absolute Pose Error (APE) 
evo_ape tum ground_truth.txt estimated_trajectory.txt --plot --save_results results/

# Generate trajectory plots
evo_traj tum ground_truth.txt estimated_trajectory.txt --plot --save_plot trajectory_comparison.pdf
```

The TUM format trajectory files generated by the application are fully compatible with EVO's evaluation metrics and visualization tools.

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
