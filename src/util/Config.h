#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace lightweight_vio {

class Config {
public:
    static Config& getInstance() {
        static Config instance;
        return instance;
    }
    
    bool load(const std::string& config_file);
    
    // Feature Detection Parameters
    int getMaxFeatures() const { return max_features; }
    double getQualityLevel() const { return quality_level; }
    double getMinDistance() const { return min_distance; }
    
    // Optical Flow Parameters
    int getWindowSize() const { return window_size; }
    int getMaxLevel() const { return max_level; }
    int getMaxIterations() const { return max_iterations; }
    double getEpsilon() const { return epsilon; }
    double getErrorThreshold() const { return error_threshold; }
    double getMaxMovement() const { return max_movement; }
    cv::TermCriteria getTermCriteria() const { 
        return cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 
                               max_iterations, epsilon); 
    }
    double getMinEigenThreshold() const { return min_eigen_threshold; }
    
    // Outlier Rejection Parameters
    double getFundamentalThreshold() const { return fundamental_threshold; }
    double getFundamentalConfidence() const { return fundamental_confidence; }
    double getMaxMovementDistance() const { return max_movement_distance; }
    double getMaxVelocityChange() const { return max_velocity_change; }
    int getMinPointsForRansac() const { return min_points_for_ransac; }
    
    // Stereo Matching Parameters
    double getStereoErrorThreshold() const { return stereo_error_threshold; }
    double getMinDisparity() const { return min_disparity; }
    double getMaxDisparity() const { return max_disparity; }
    double getMaxYDifference() const { return max_y_difference; }
    double getEpipolarThreshold() const { return epipolar_threshold; }
    
    // Stereo Rectification Parameters
    double getMaxRectifiedYDifference() const { return max_rectified_y_difference; }
    
    // Global Depth Parameters (used throughout VIO system)
    double getMinDepth() const { return min_depth; }
    double getMaxDepth() const { return max_depth; }
    
    // Triangulation Parameters (specific to 3D point generation)
    double getMaxReprojectionError() const { return max_reprojection_error; }
    double getMinParallax() const { return min_parallax; }
    
    // Keyframe Parameters
    int getKeyframeInterval() const { return keyframe_interval; }
    
    // Camera Parameters
    int getImageWidth() const { return image_width; }
    int getImageHeight() const { return image_height; }
    int getBorderSize() const { return border_size; }
    
    // Camera Calibration
    cv::Mat getLeftCameraMatrix() const { return left_camera_matrix.clone(); }
    cv::Mat getRightCameraMatrix() const { return right_camera_matrix.clone(); }
    cv::Mat getLeftDistCoeffs() const { return left_dist_coeffs.clone(); }
    cv::Mat getRightDistCoeffs() const { return right_dist_coeffs.clone(); }
    cv::Mat getLeftToRightTransform() const { return T_left_right.clone(); }  // T_rl: left to right transform
    
    // Convenient camera parameter getters
    double getFx() const { return left_camera_matrix.at<double>(0, 0); }
    double getFy() const { return left_camera_matrix.at<double>(1, 1); }
    double getCx() const { return left_camera_matrix.at<double>(0, 2); }
    double getCy() const { return left_camera_matrix.at<double>(1, 2); }
    double getBaseline() const { return 0.11; }  // EuRoC stereo baseline (meters)
    cv::Mat getDistortionCoeffs() const { return left_dist_coeffs.clone(); }
    
    // Performance Parameters
    bool isTimingEnabled() const { return enable_timing; }
    bool isDebugOutputEnabled() const { return enable_debug_output; }

private:
    Config() = default;
    
    // Feature Detection Parameters
    int max_features = 150;
    double quality_level = 0.01;
    double min_distance = 30.0;
    
    // Optical Flow Parameters
    int window_size = 21;
    int max_level = 3;
    int max_iterations = 30;
    double epsilon = 0.01;
    double error_threshold = 30.0;
    double max_movement = 100.0;
    double min_eigen_threshold = 1e-4;
    
    // Outlier Rejection Parameters
    double fundamental_threshold = 1.0;
    double fundamental_confidence = 0.99;
    double max_movement_distance = 50.0;
    double max_velocity_change = 20.0;
    int min_points_for_ransac = 8;
    
    // Stereo Matching Parameters
    double stereo_error_threshold = 50.0;
    double min_disparity = 0.1;
    double max_disparity = 300.0;
    double max_y_difference = 20.0;
    double epipolar_threshold = 5.0;
    
    // Stereo Rectification Parameters
    double max_rectified_y_difference = 2.0;
    
    // Global Depth Parameters (used throughout VIO system)
    double min_depth = 0.1;
    double max_depth = 100.0;
    
    // Triangulation Parameters (specific to 3D point generation)
    double max_reprojection_error = 2.0;
    double min_parallax = 0.5;
    
    // Keyframe Parameters
    int keyframe_interval = 10;
    
    // Camera Parameters
    int image_width = 752;
    int image_height = 480;
    int border_size = 1;
    
    // Camera Calibration Matrices
    cv::Mat left_camera_matrix;
    cv::Mat right_camera_matrix;
    cv::Mat left_dist_coeffs;
    cv::Mat right_dist_coeffs;
    cv::Mat T_left_right;  // T_rl: Transform from left to right camera (following T_ab = b->a convention)
    
    // Performance Parameters
    bool enable_timing = true;
    bool enable_debug_output = true;
};

} // namespace lightweight_vio
