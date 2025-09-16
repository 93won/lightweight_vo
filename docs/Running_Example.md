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
