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
