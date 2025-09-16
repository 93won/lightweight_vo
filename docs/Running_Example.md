# Running Examples

This guide provides detailed instructions for running the Lightweight Stereo VIO system with the EuRoC dataset.

## Quick Start

### Native Build
```bash
# Visual Odometry (VO) mode
./build/euroc_stereo config/euroc_vo.yaml /path/to/EuRoC/MH_01_easy

# Visual-Inertial Odometry (VIO) mode
./build/euroc_stereo config/euroc_vio.yaml /path/to/EuRoC/MH_01_easy
```

### Docker
```bash
# Visual Odometry (VO) mode
./docker.sh run vo /path/to/EuRoC/MH_01_easy

# Visual-Inertial Odometry (VIO) mode
./docker.sh run vio /path/to/EuRoC/MH_01_easy
```

---

## Detailed Usage

### Command Line Interface

The general syntax for running the application is:

```bash
./build/euroc_stereo <config_file_path> <euroc_dataset_path>
```

#### Parameters:
- `<config_file_path>`: Path to the YAML configuration file
- `<euroc_dataset_path>`: Path to a specific EuRoC sequence directory

### Configuration Files

The system behavior is controlled by YAML configuration files located in the `config/` directory:

#### Visual Odometry (VO) Mode
**File**: `config/euroc_vo.yaml`
- Uses only camera data
- Suitable for scenarios with good visual features
- Lower computational requirements

#### Visual-Inertial Odometry (VIO) Mode  
**File**: `config/euroc_vio.yaml`
- Uses both camera and IMU data
- More robust to motion blur and challenging conditions
- Higher computational requirements but better accuracy

### Mode Selection

The system mode is automatically determined by the `system_mode.mode` setting in the YAML configuration file:
- `euroc_vo.yaml` contains: `system_mode.mode: "VO"`
- `euroc_vio.yaml` contains: `system_mode.mode: "VIO"`

---

## Native Build Examples

### Prerequisites
- Complete installation following [Install.md](Install.md)
- Download dataset following [Download_Dataset.md](Download_Dataset.md)

### Example Commands

#### Easy Sequences (Recommended for first run)
```bash
# MH_01_easy with VO
./build/euroc_stereo config/euroc_vo.yaml /path/to/EuRoC/MH_01_easy

# MH_01_easy with VIO  
./build/euroc_stereo config/euroc_vio.yaml /path/to/EuRoC/MH_01_easy

# V1_01_easy with VIO
./build/euroc_stereo config/euroc_vio.yaml /path/to/EuRoC/V1_01_easy
```

#### Medium Difficulty Sequences
```bash
# MH_03_medium with VIO
./build/euroc_stereo config/euroc_vio.yaml /path/to/EuRoC/MH_03_medium

# V1_02_medium with VIO
./build/euroc_stereo config/euroc_vio.yaml /path/to/EuRoC/V1_02_medium
```

#### Challenging Sequences
```bash
# MH_04_difficult with VIO (recommended for VIO due to fast motion)
./build/euroc_stereo config/euroc_vio.yaml /path/to/EuRoC/MH_04_difficult
```

---

## Docker Examples

### Prerequisites
- Complete Docker installation following [Install.md](Install.md)
- Download dataset following [Download_Dataset.md](Download_Dataset.md)

### X11 Setup for Visualization

Before running Docker containers, ensure X11 forwarding is properly configured:

```bash
# Allow local connections to X server
xhost +local:docker
```

### Example Commands

#### Easy Sequences
```bash
# MH_01_easy with VO
./docker.sh run vo /path/to/EuRoC/MH_01_easy

# MH_01_easy with VIO
./docker.sh run vio /path/to/EuRoC/MH_01_easy

# V1_01_easy with VIO
./docker.sh run vio /path/to/EuRoC/V1_01_easy
```

#### Medium Difficulty Sequences
```bash
# MH_03_medium with VIO
./docker.sh run vio /path/to/EuRoC/MH_03_medium

# V1_02_medium with VIO  
./docker.sh run vio /path/to/EuRoC/V1_02_medium
```

#### Challenging Sequences
```bash
# MH_04_difficult with VIO
./docker.sh run vio /path/to/EuRoC/MH_04_difficult
```

### Docker Run Details

The `docker.sh run` command:
- Mounts the dataset directory as read-only
- Forwards X11 display for Pangolin viewer
- Automatically selects the appropriate config file
- Removes the container after execution

---

## Understanding the Output

### Real-time Visualization

During execution, you'll see:
- **Pangolin Viewer**: 3D visualization showing:
  - Camera trajectory (green line)
  - Current camera pose
  - Detected features
  - IMU data visualization (VIO mode)

### Console Output

The application displays:
- Frame processing progress
- Timing information
- Feature tracking statistics
- Optimization convergence info

### Output Files

After completion, the following files are generated:

#### Trajectory Files
- `estimated_trajectory_vo.txt` or `estimated_trajectory_vio.txt`
- `ground_truth_vo.txt` or `ground_truth_vio.txt`
- Format: TUM trajectory format (timestamp tx ty tz qx qy qz qw)

#### Statistics Files
- `statistics_vo.txt` or `statistics_vio.txt`
- Contains detailed performance analysis

---

## Performance Analysis

### Built-in Analysis Features

The application automatically analyzes performance and outputs:

#### 1. Timing Analysis
- Frame processing times
- Average frame rate (fps)
- Processing time statistics

#### 2. Velocity Analysis  
- Linear velocity (m/s)
- Angular velocity (rad/s)
- Statistical metrics (mean, median, min, max)

#### 3. Transform Error Analysis
- Frame-to-frame pose errors
- Rotation and translation error statistics
- RMSE calculations

### Example Statistics Output

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

               FRAME-TO-FRAME TRANSFORM ERROR ANALYSIS              
════════════════════════════════════════════════════════════════════
 Total Frame Pairs Analyzed: 3637
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

---

## External Evaluation with EVO

For comprehensive trajectory evaluation, use the [EVO tool](https://github.com/MichaelGrupp/evo):

### Installation
```bash
pip install evo
```

### Usage Examples
```bash
# Relative Pose Error (RPE) analysis
evo_rpe tum ground_truth.txt estimated_trajectory.txt --plot --save_results results/

# Absolute Pose Error (APE) analysis  
evo_ape tum ground_truth.txt estimated_trajectory.txt --plot --save_results results/

# Trajectory visualization
evo_traj tum ground_truth.txt estimated_trajectory.txt --plot --save_plot trajectory_comparison.pdf
```

---

## Troubleshooting

### Common Issues

#### Pangolin Viewer Not Displaying
```bash
# For Docker: Check X11 forwarding
xhost +local:docker
echo $DISPLAY

# For native build: Install OpenGL drivers
sudo apt install mesa-utils
glxinfo | grep "direct rendering"
```

#### Dataset Path Issues
```bash
# Verify dataset structure
ls -la /path/to/EuRoC/MH_01_easy/mav0/
# Should show: cam0/, cam1/, imu0/, leica0/, etc.
```

#### Performance Issues
- Use VIO mode for fast motion sequences
- Reduce image resolution in config if needed
- Check system resources (CPU/memory usage)

#### Configuration Errors
- Verify YAML syntax in config files
- Check file paths in configuration
- Ensure all required parameters are set

### Getting Help

If you encounter issues:
1. Check the console output for error messages
2. Verify your dataset and configuration files
3. Test with the recommended easy sequences first
4. Check system requirements and dependencies

---

## Advanced Usage

### Custom Configuration

You can create custom configuration files by modifying the existing ones:

```bash
cp config/euroc_vio.yaml config/my_custom_config.yaml
# Edit my_custom_config.yaml as needed
./build/euroc_stereo config/my_custom_config.yaml /path/to/dataset
```

### Batch Processing

Process multiple sequences:

```bash
#!/bin/bash
sequences=("MH_01_easy" "MH_02_easy" "MH_03_medium")
for seq in "${sequences[@]}"; do
    ./build/euroc_stereo config/euroc_vio.yaml /path/to/EuRoC/$seq
done
```

### Real-time Performance Tips

For optimal real-time performance:
- Use Release build configuration
- Disable unnecessary logging
- Adjust thread counts in configuration
- Consider CPU affinity settings
