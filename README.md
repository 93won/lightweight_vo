# Lightweight Stereo VIO

This is a lightweight stereo visual-inertial odometry (VIO) project designed for real-time performance. It utilizes feature tracking, IMU pre-integration, sliding window optimization with Ceres Solver, and Pangolin for visualization.

## Features

- Stereo vision-based visual-inertial odometry
- IMU pre-integration for robust state estimation
- Sliding window optimization using Ceres Solver
- Real-time visualization with Pangolin
- Docker support for easy deployment and testing

## Demo
[![Stereo VO Demo](https://img.youtube.com/vi/41o9R-rKQ1s/0.jpg)](https://www.youtube.com/watch?v=41o9R-rKQ1s)

## To-Do List

- [x] **IMU Integration**: Incorporate IMU data to create a more robust Visual-Inertial Odometry (VIO) system, improving accuracy and handling of fast motions.
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

The provided `build.sh` script will automatically install system dependencies (like OpenCV, Eigen, etc.) and then build the third-party libraries (Ceres, Pangolin) and the main application.

```bash
chmod +x build.sh
./build.sh
```

### Step 1-3: Prepare the EuRoC Dataset


First, download the dataset on your host machine. The repository includes a convenience script to download all EuRoC MAV datasets (11 sequences total).

```bash
chmod +x script/download_euroc.sh
./script/download_euroc.sh /path/of/dataset
```
This will download all sequences into a `/path/of/dataset` directory at the root of the project. This directory will be mounted into the Docker container.

### Step 1-4: Run the Application

After the build is complete, you can run the VO or VIO with a EuRoC dataset. You need to provide the configuration file path and the dataset path as arguments.

#### Running the Visual Odometry (VO)
```bash
./build/euroc_stereo_vo <config_file_path> <euroc_dataset_path>
```

**Example:**
```bash
./build/euroc_stereo_vo /home/lightweight_vo/config/euroc_vo.yaml /path/to/your/EuRoC/MH_01_easy
```

#### Running the Visual-Inertial Odometry (VIO)
```bash
./build/euroc_stereo_vio <config_file_path> <euroc_dataset_path>
```

**Example:**
```bash
./build/euroc_stereo_vio /home/lightweight_vo/config/euroc_vio.yaml /path/to/your/EuRoC/MH_01_easy
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

### Step 2-3: Run the Application in a Container

This command runs the application inside a new container. It mounts your local dataset directory into the container (read-only) and forwards your X11 display for the Pangolin viewer.

You can specify `vo` or `vio` as the first argument to choose which program to run.

#### Running the VO
```bash
./docker.sh run vo /path/of/dataset/EuRoC/MH_01_easy
```

#### Running the VIO
```bash
./docker.sh run vio /path/of/dataset/EuRoC/MH_01_easy
```

The container will automatically execute the VIO with the built-in config file and the provided dataset.

## Performance Analysis and Evaluation

The application automatically performs a comprehensive analysis of the VIO results and outputs detailed statistics to the console and a file (`statistics_vio.txt`) upon completion.

### 1. Built-in Analysis Features

**1-1. Timing Analysis** 
- Measures frame processing times throughout the entire sequence.
- Reports average processing time (ms) and frame rate (fps).

**1-2. Velocity Analysis**
- Calculates the linear (m/s) and angular (rad/s) velocities between consecutive frames.
- Provides statistical metrics: mean, median, min, and max.

**1-3. Frame-to-Frame Transform Error Analysis**
- Compares the relative pose between consecutive frames against the ground truth.
- Provides statistical metrics for rotation (°) and translation (m) errors: mean, median, min, max, and RMSE.

**1-4. Trajectory Output**
- Saves both the estimated and ground truth trajectories in TUM format (`estimated_trajectory_vio.txt`, `ground_truth_vio.txt`).
- These files are fully compatible with external evaluation tools like EVO.

### 2. Example Output

Here is an example of the statistics file generated by the application:

```
════════════════════════════════════════════════════════════════════
                          STATISTICS (VIO)                          
════════════════════════════════════════════════════════════════════

                          TIMING ANALYSIS                           
════════════════════════════════════════════════════════════════════
 Total Frames Processed: 3638
 Average Processing Time: 10.35ms
 Average Frame Rate: 96.7fps

                          VELOCITY ANALYSIS                         
════════════════════════════════════════════════════════════════════
                        LINEAR VELOCITY (m/s)                       
 Mean      :     0.4334m/s
 Median    :     0.4343m/s
 Minimum   :     0.0000m/s
 Maximum   :     2.0516m/s

                       ANGULAR VELOCITY (rad/s)                     
 Mean      :     0.1803rad/s
 Median    :     0.1377rad/s
 Minimum   :     0.0001rad/s
 Maximum   :     0.9426rad/s

               FRAME-TO-FRAME TRANSFORM ERROR ANALYSIS              
════════════════════════════════════════════════════════════════════
 Total Frame Pairs Analyzed: 3637 (all_frames: 3638, gt_poses: 3638)
 Frame precision: 32 bit floats

                     ROTATION ERROR STATISTICS                    
 Mean      :     0.0516°
 Median    :     0.0494°
 Minimum   :     0.0003°
 Maximum   :     0.1808°
 RMSE      :     0.0609°

                   TRANSLATION ERROR STATISTICS                   
 Mean      :   0.001734m
 Median    :   0.001518m
 Minimum   :   0.000006m
 Maximum   :   0.008671m
 RMSE      :   0.002200m

════════════════════════════════════════════════════════════════════
```

### 3. External Evaluation with EVO

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

- `app/`: Main application entry points (`euroc_stereo_vo.cpp`, `euroc_stereo_vio.cpp`).
- `src/`:
  - `database/`: Data structures for `Frame` (including IMU data), `MapPoint`, and `Feature`.
  - `processing/`: Core VIO modules, including `Estimator`, `FeatureTracker`, `IMUHandler` (pre-integration), and `Optimizer` (sliding window optimization).
  - `optimization/`: Ceres Solver cost functions (`Factors`) for visual reprojection errors and IMU pre-integration constraints.
  - `viewer/`: Pangolin-based visualization.
  - `util/`: Utility functions for configuration and data loading.
- `thirdparty/`: External libraries (Ceres, Pangolin, Sophus, spdlog).
- `config/`: Configuration files for VO and VIO.
- `scripts/`: Helper scripts for downloading datasets and running Docker.
