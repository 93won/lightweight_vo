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
    
    // Outlier Rejection Parameters
    double getFundamentalThreshold() const { return fundamental_threshold; }
    double getMaxMovementDistance() const { return max_movement_distance; }
    double getMaxVelocityChange() const { return max_velocity_change; }
    int getMinPointsForRansac() const { return min_points_for_ransac; }
    
    // Stereo Matching Parameters
    double getStereoErrorThreshold() const { return stereo_error_threshold; }
    double getMinDisparity() const { return min_disparity; }
    double getMaxDisparity() const { return max_disparity; }
    double getMaxYDifference() const { return max_y_difference; }
    double getEpipolarThreshold() const { return epipolar_threshold; }
    
    // Camera Parameters
    int getImageWidth() const { return image_width; }
    int getImageHeight() const { return image_height; }
    int getBorderSize() const { return border_size; }
    
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
    
    // Outlier Rejection Parameters
    double fundamental_threshold = 1.0;
    double max_movement_distance = 50.0;
    double max_velocity_change = 20.0;
    int min_points_for_ransac = 8;
    
    // Stereo Matching Parameters
    double stereo_error_threshold = 50.0;
    double min_disparity = 0.1;
    double max_disparity = 300.0;
    double max_y_difference = 20.0;
    double epipolar_threshold = 5.0;
    
    // Camera Parameters
    int image_width = 752;
    int image_height = 480;
    int border_size = 1;
    
    // Performance Parameters
    bool enable_timing = true;
    bool enable_debug_output = true;
};

} // namespace lightweight_vio
