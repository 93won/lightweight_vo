# Performance Analysis and Evaluation

The Lightweight Stereo VIO system provides comprehensive performance analysis and evaluation capabilities to help users understand system behavior and validate results against ground truth data.

## Built-in Analysis Features

The application automatically performs comprehensive analysis and outputs detailed statistics upon completion, including:

- **Timing Analysis**: Processing times and frame rates
- **Velocity Analysis**: Linear and angular velocity statistics  
- **Transform Error Analysis**: Frame-to-frame pose error metrics
- **Trajectory Output**: TUM format files compatible with [EVO](https://github.com/MichaelGrupp/evo) evaluation tool

### 1. Timing Analysis

Measures performance characteristics throughout the processing:

- **Frame Processing Times**: Individual frame processing duration
- **Average Processing Time**: Mean processing time per frame (ms)
- **Frame Rate**: Average frames per second (fps)
- **Total Processing Time**: Complete sequence processing duration

### 2. Velocity Analysis

Calculates motion characteristics between consecutive frames:

#### Linear Velocity Analysis
- Translational motion between frames (m/s)
- Statistical metrics: mean, median, minimum, maximum
- Useful for understanding trajectory smoothness

#### Angular Velocity Analysis  
- Rotational motion between frames (rad/s)
- Statistical metrics: mean, median, minimum, maximum
- Indicates camera rotation patterns and stability

### 3. Transform Error Analysis

Compares estimated poses against ground truth:

#### Frame-to-Frame Error Metrics
- **Rotation Error**: Angular difference between consecutive estimated and ground truth poses (degrees)
- **Translation Error**: Euclidean distance between consecutive estimated and ground truth positions (meters)
- **Statistical Analysis**: Mean, median, minimum, maximum, and RMSE for both rotation and translation errors

#### Error Calculation Method
- Relative pose computation between consecutive frames
- Comparison with ground truth relative transformations
- Error metrics calculated in pose manifold space

### 4. Trajectory Output

Generates standard format trajectory files for external analysis:

#### File Formats
- **TUM Format**: `timestamp tx ty tz qx qy qz qw`
- **Estimated Trajectory**: `estimated_trajectory_vio.txt` or `estimated_trajectory_vo.txt`
- **Ground Truth**: `ground_truth_vio.txt` or `ground_truth_vo.txt`

#### Compatibility
- Full compatibility with [EVO](https://github.com/MichaelGrupp/evo) evaluation toolkit
- Standard robotics trajectory format
- Easy integration with custom analysis tools

## Example Analysis Output

### Console Statistics Display

The system outputs detailed statistics to both console and file upon completion:

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

### Output Files Generated

After processing completion, the following files are automatically created:

#### Statistics Files
- `statistics_vio.txt` (VIO mode) or `statistics_vo.txt` (VO mode)
- Contains the same detailed analysis shown in console
- Formatted for easy parsing and archival

#### Trajectory Files
- `estimated_trajectory_vio.txt` / `estimated_trajectory_vo.txt`
- `ground_truth_vio.txt` / `ground_truth_vo.txt`
- TUM format: `timestamp tx ty tz qx qy qz qw`

## External Evaluation with EVO

For comprehensive trajectory evaluation beyond built-in analysis, use the [EVO (Python package for the evaluation of odometry and SLAM)](https://github.com/MichaelGrupp/evo) toolkit.

### EVO Installation

```bash
pip install evo
```

### EVO Usage Examples

#### Relative Pose Error (RPE) Analysis
```bash
# Analyze frame-to-frame pose errors
evo_rpe tum ground_truth_vio.txt estimated_trajectory_vio.txt --plot --save_results results/

# RPE with specific delta (e.g., 1.0 second intervals)
evo_rpe tum ground_truth_vio.txt estimated_trajectory_vio.txt --delta 1.0 --plot
```

#### Absolute Pose Error (APE) Analysis
```bash
# Analyze absolute trajectory errors
evo_ape tum ground_truth_vio.txt estimated_trajectory_vio.txt --plot --save_results results/

# APE with alignment (removes scale, rotation, translation bias)
evo_ape tum ground_truth_vio.txt estimated_trajectory_vio.txt --align --plot
```

#### Trajectory Visualization
```bash
# Compare trajectories visually
evo_traj tum ground_truth_vio.txt estimated_trajectory_vio.txt --plot --save_plot trajectory_comparison.pdf

# 3D trajectory plot
evo_traj tum ground_truth_vio.txt estimated_trajectory_vio.txt --plot_mode xyz --save_plot trajectory_3d.pdf
```

#### Advanced Analysis
```bash
# Generate comprehensive report
evo_res results/*.zip --save_table results_table.csv --plot --save_plot results_comparison.pdf

# Multi-sequence comparison
evo_ape tum gt1.txt est1.txt gt2.txt est2.txt --plot --save_results results/
```

### EVO Output Interpretation

#### RPE (Relative Pose Error)
- **Local Accuracy**: How accurate is the system over short time intervals
- **Drift Analysis**: Identifies systematic drift patterns
- **Motion-dependent Errors**: Correlates errors with motion characteristics

#### APE (Absolute Pose Error)  
- **Global Accuracy**: Overall trajectory accuracy
- **Scale Estimation**: How well the system estimates metric scale
- **Loop Closure Performance**: Accuracy after revisiting locations
