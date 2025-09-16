# Lightweight Stereo VIO

This is a lightweight stereo visual-inertial odometry (VIO) project designed for real-time performance. It utilizes feature tracking, IMU pre-integration, sliding window optimization with Ceres Solver, and Pangolin for visualization.

## Features

- Stereo vision-based visual-inertial odometry
- IMU pre-integration for robust state estimation
- Sliding window optimization using Ceres Solver
- Real-time visualization with Pangolin
- Docker support for easy deployment and testing

## Demo
[![Stereo VIO Demo](https://img.youtube.com/vi/qdnn4ShEpTA/0.jpg)](https://youtu.be/qdnn4ShEpTA)

## To-Do List

- [x] **IMU Integration**: Incorporate IMU data to create a more robust Visual-Inertial Odometry (VIO) system, improving accuracy and handling of fast motions.
- [ ] **Fisheye Camera Support**: Add support for fisheye camera models and distortion correction for wide-angle stereo setups.
- [ ] **Embedded Environment Testing**: Test and optimize performance on embedded platforms like Jetson Nano, NUC, and other resource-constrained devices.

## Installation

📋 **[Installation Guide](docs/Install.md)** - Complete installation instructions for both Docker and native builds

Choose between Docker (recommended) or native Ubuntu 22.04 build. Includes prerequisites, dependencies, and troubleshooting.

## Dataset Download

📁 **[Dataset Download Guide](docs/Download_Dataset.md)** - EuRoC dataset download and preparation

Automated scripts and manual download options for the EuRoC MAV dataset. Includes dataset structure and sequence recommendations.

## Running the Application

🚀 **[Running Examples](docs/Running_Example.md)** - Usage examples and performance analysis

Comprehensive guide for running VIO/VO modes with Docker or native builds. Includes output analysis and EVO evaluation.

## Performance Analysis and Evaluation

The application automatically performs comprehensive analysis and outputs detailed statistics upon completion, including:

- **Timing Analysis**: Processing times and frame rates
- **Velocity Analysis**: Linear and angular velocity statistics
- **Transform Error Analysis**: Frame-to-frame pose error metrics
- **Trajectory Output**: TUM format files compatible with [EVO](https://github.com/MichaelGrupp/evo) evaluation tool

### Example Output
```
[2025-09-16 14:14:28.685] [info] ════════════════════════════════════════════════════════════════════
[2025-09-16 14:14:28.685] [info]                           STATISTICS (VIO)                          
[2025-09-16 14:14:28.685] [info] ════════════════════════════════════════════════════════════════════
[2025-09-16 14:14:28.685] [info] 
[2025-09-16 14:14:28.685] [info]                           TIMING ANALYSIS                           
[2025-09-16 14:14:28.685] [info] ════════════════════════════════════════════════════════════════════
[2025-09-16 14:14:28.685] [info]  Total Frames Processed: 2221
[2025-09-16 14:14:28.685] [info]  Average Processing Time: 9.29ms
[2025-09-16 14:14:28.685] [info]  Average Frame Rate: 107.7fps
[2025-09-16 14:14:28.685] [info] 
[2025-09-16 14:14:28.685] [info]                           VELOCITY ANALYSIS                         
[2025-09-16 14:14:28.685] [info] ════════════════════════════════════════════════════════════════════
[2025-09-16 14:14:28.685] [info]                         LINEAR VELOCITY (m/s)                       
[2025-09-16 14:14:28.685] [info]  Mean      :     0.8702m/s
[2025-09-16 14:14:28.685] [info]  Median    :     0.8914m/s
[2025-09-16 14:14:28.685] [info]  Minimum   :     0.0001m/s
[2025-09-16 14:14:28.685] [info]  Maximum   :     3.0937m/s
[2025-09-16 14:14:28.685] [info] 
[2025-09-16 14:14:28.685] [info]                        ANGULAR VELOCITY (rad/s)                     
[2025-09-16 14:14:28.685] [info]  Mean      :     0.1780rad/s
[2025-09-16 14:14:28.685] [info]  Median    :     0.1217rad/s
[2025-09-16 14:14:28.685] [info]  Minimum   :     0.0004rad/s
[2025-09-16 14:14:28.685] [info]  Maximum   :     1.1003rad/s
[2025-09-16 14:14:28.685] [info] 
[2025-09-16 14:14:28.685] [info]                FRAME-TO-FRAME TRANSFORM ERROR ANALYSIS              
[2025-09-16 14:14:28.685] [info] ════════════════════════════════════════════════════════════════════
[2025-09-16 14:14:28.685] [info]  Total Frame Pairs Analyzed: 2220 (all_frames: 2221, gt_poses: 2221)
[2025-09-16 14:14:28.685] [info]  Frame precision: 32 bit floats
[2025-09-16 14:14:28.685] [info] 
[2025-09-16 14:14:28.685] [info]                      ROTATION ERROR STATISTICS                    
[2025-09-16 14:14:28.685] [info]  Mean      :     0.0851°
[2025-09-16 14:14:28.685] [info]  Median    :     0.0803°
[2025-09-16 14:14:28.685] [info]  Minimum   :     0.0007°
[2025-09-16 14:14:28.685] [info]  Maximum   :     0.2947°
[2025-09-16 14:14:28.685] [info]  RMSE      :     0.0997°
[2025-09-16 14:14:28.685] [info] 
[2025-09-16 14:14:28.685] [info]                    TRANSLATION ERROR STATISTICS                   
[2025-09-16 14:14:28.685] [info]  Mean      :   0.004201m
[2025-09-16 14:14:28.685] [info]  Median    :   0.003513m
[2025-09-16 14:14:28.685] [info]  Minimum   :   0.000011m
[2025-09-16 14:14:28.685] [info]  Maximum   :   0.053683m
[2025-09-16 14:14:28.685] [info]  RMSE      :   0.005488m
[2025-09-16 14:14:28.685] [info] ════════════════════════════════════════════════════════════════════
[2025-09-16 14:14:28.685] [info] [EurocPlayer] Processing completed! Click 'Finish & Exit' to close.
```

See [Running Examples](docs/Running_Example.md) for complete output analysis details.

## Project Structure

The source code is organized into the following directories:

- `app/`: Main application entry points
- `src/`:
  - `database/`: Data structures for `Frame` (including IMU data), `MapPoint`, and `Feature`
  - `processing/`: Core VIO modules, including `Estimator`, `FeatureTracker`, `IMUHandler`, and `Optimizer`
  - `optimization/`: Ceres Solver cost functions for visual reprojection errors and IMU pre-integration constraints
  - `viewer/`: Pangolin-based visualization
  - `util/`: Utility functions for configuration and data loading
- `thirdparty/`: External libraries (Ceres, Pangolin, Sophus, spdlog)
- `config/`: Configuration files for VO and VIO modes
- `docs/`: Detailed documentation guides

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{lightweight_vo,
  title={Lightweight Stereo Visual-Inertial Odometry},
  author={Your Name},
  year={2024},
  url={https://github.com/93won/lightweight_vo}
}
```
