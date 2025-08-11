#include "Config.h"
#include <opencv2/opencv.hpp>
#include <iostream>

namespace lightweight_vio {

bool Config::load(const std::string& config_file) {
    cv::FileStorage fs(config_file, cv::FileStorage::READ);
    
    if (!fs.isOpened()) {
        std::cerr << "Failed to open config file: " << config_file << std::endl;
        return false;
    }
    
    // Feature Detection Parameters
    cv::FileNode feature_detection = fs["feature_detection"];
    if (!feature_detection.empty()) {
        max_features = (int)feature_detection["max_features"];
        quality_level = (double)feature_detection["quality_level"];
        min_distance = (double)feature_detection["min_distance"];
    }
    
    // Optical Flow Parameters
    cv::FileNode optical_flow = fs["optical_flow"];
    if (!optical_flow.empty()) {
        window_size = (int)optical_flow["window_size"];
        max_level = (int)optical_flow["max_level"];
        max_iterations = (int)optical_flow["max_iterations"];
        epsilon = (double)optical_flow["epsilon"];
        error_threshold = (double)optical_flow["error_threshold"];
        max_movement = (double)optical_flow["max_movement"];
    }
    
    // Outlier Rejection Parameters
    cv::FileNode outlier_rejection = fs["outlier_rejection"];
    if (!outlier_rejection.empty()) {
        fundamental_threshold = (double)outlier_rejection["fundamental_threshold"];
        max_movement_distance = (double)outlier_rejection["max_movement_distance"];
        max_velocity_change = (double)outlier_rejection["max_velocity_change"];
        min_points_for_ransac = (int)outlier_rejection["min_points_for_ransac"];
    }
    
    // Stereo Matching Parameters
    cv::FileNode stereo_matching = fs["stereo_matching"];
    if (!stereo_matching.empty()) {
        stereo_error_threshold = (double)stereo_matching["error_threshold"];
        min_disparity = (double)stereo_matching["min_disparity"];
        max_disparity = (double)stereo_matching["max_disparity"];
        max_y_difference = (double)stereo_matching["max_y_difference"];
        epipolar_threshold = (double)stereo_matching["epipolar_threshold"];
    }
    
    // Camera Parameters
    cv::FileNode camera = fs["camera"];
    if (!camera.empty()) {
        image_width = (int)camera["image_width"];
        image_height = (int)camera["image_height"];
        border_size = (int)camera["border_size"];
    }
    
    // Performance Parameters
    cv::FileNode performance = fs["performance"];
    if (!performance.empty()) {
        enable_timing = (bool)(int)performance["enable_timing"];
        enable_debug_output = (bool)(int)performance["enable_debug_output"];
    }
    
    fs.release();
    
    std::cout << "Loaded configuration from: " << config_file << std::endl;
    std::cout << "Max features: " << max_features << std::endl;
    std::cout << "Window size: " << window_size << std::endl;
    std::cout << "Fundamental threshold: " << fundamental_threshold << std::endl;
    
    return true;
}

} // namespace lightweight_vio
