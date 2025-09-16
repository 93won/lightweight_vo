# Lightweight Stereo VIO

This is a lightweight stereo visual-inertial odometry (VIO) project designed for real-time performance. It utilizes feature tracking, IMU pre-integration, sliding window optimization with Ceres Solver, and Pangolin for visualization.

## Demo
[![Stereo VIO Demo](https://img.youtube.com/vi/qdnn4ShEpTA/0.jpg)](https://youtu.be/qdnn4ShEpTA)

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

📊 **[Performance Analysis Guide](docs/Performance_Analysis.md)** - Comprehensive performance evaluation and benchmarking

Built-in timing, velocity, and error analysis with TUM format output. Includes EVO evaluation toolkit integration and benchmarking guidelines.

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

## References

This project's IMU pre-integration implementation is based on the following paper:

```bibtex
@article{forster2016manifold,
  author = {Forster, Christian and Carlone, Luca and Dellaert, Frank and Scaramuzza, Davide},
  year = {2016},
  month = {08},
  title = {On-Manifold Preintegration for Real-Time Visual-Inertial Odometry},
  volume = {33},
  journal = {IEEE Transactions on Robotics},
  doi = {10.1109/TRO.2016.2597321}
}
```



