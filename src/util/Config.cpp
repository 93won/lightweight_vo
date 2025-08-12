#include "Config.h"
#include <opencv2/opencv.hpp>
#include <iostream>

namespace lightweight_vio {

bool Config::load(const std::string& config_file) {
    std::cout << "[DEBUG] Trying to load config file: " << config_file << std::endl;
    
    cv::FileStorage fs(config_file, cv::FileStorage::READ);
    
    if (!fs.isOpened()) {
        std::cerr << "Failed to open config file: " << config_file << std::endl;
        return false;
    }
    
    std::cout << "[DEBUG] Config file opened successfully" << std::endl;
    
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
        min_eigen_threshold = (double)optical_flow["min_eigen_threshold"];
    }
    
    // Outlier Rejection Parameters
    cv::FileNode outlier_rejection = fs["outlier_rejection"];
    if (!outlier_rejection.empty()) {
        fundamental_threshold = (double)outlier_rejection["fundamental_threshold"];
        fundamental_confidence = (double)outlier_rejection["fundamental_confidence"];
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
    
    // Triangulation Parameters
    cv::FileNode triangulation = fs["triangulation"];
    if (!triangulation.empty()) {
        min_depth = (double)triangulation["min_depth"];
        max_depth = (double)triangulation["max_depth"];
        max_reprojection_error = (double)triangulation["max_reprojection_error"];
        min_parallax = (double)triangulation["min_parallax"];
    }
    
    // Camera Parameters
    cv::FileNode camera = fs["camera"];
    if (!camera.empty()) {
        std::cout << "Loading camera parameters..." << std::endl;
        image_width = (int)camera["image_width"];
        image_height = (int)camera["image_height"];
        border_size = (int)camera["border_size"];
        
        // Load camera intrinsics
        cv::FileNode left_intrinsics = camera["left_intrinsics"];
        cv::FileNode right_intrinsics = camera["right_intrinsics"];
        cv::FileNode left_distortion = camera["left_distortion"];
        cv::FileNode right_distortion = camera["right_distortion"];
        
        std::cout << "Left intrinsics size: " << left_intrinsics.size() << std::endl;
        std::cout << "Right intrinsics size: " << right_intrinsics.size() << std::endl;
        
        if (!left_intrinsics.empty() && left_intrinsics.size() == 4) {
            left_camera_matrix = (cv::Mat_<double>(3, 3) << 
                (double)left_intrinsics[0], 0, (double)left_intrinsics[2],
                0, (double)left_intrinsics[1], (double)left_intrinsics[3],
                0, 0, 1);
            std::cout << "Left camera matrix loaded successfully" << std::endl;
        } else {
            std::cout << "Failed to load left intrinsics" << std::endl;
        }
        
        if (!right_intrinsics.empty() && right_intrinsics.size() == 4) {
            right_camera_matrix = (cv::Mat_<double>(3, 3) << 
                (double)right_intrinsics[0], 0, (double)right_intrinsics[2],
                0, (double)right_intrinsics[1], (double)right_intrinsics[3],
                0, 0, 1);
            std::cout << "Right camera matrix loaded successfully" << std::endl;
        } else {
            std::cout << "Failed to load right intrinsics" << std::endl;
        }
        
        if (!left_distortion.empty() && left_distortion.size() == 4) {
            left_dist_coeffs = (cv::Mat_<double>(1, 4) << 
                (double)left_distortion[0], (double)left_distortion[1],
                (double)left_distortion[2], (double)left_distortion[3]);
            std::cout << "Left distortion coefficients loaded successfully" << std::endl;
        } else {
            std::cout << "Failed to load left distortion coefficients" << std::endl;
        }
        
        if (!right_distortion.empty() && right_distortion.size() == 4) {
            right_dist_coeffs = (cv::Mat_<double>(1, 4) << 
                (double)right_distortion[0], (double)right_distortion[1],
                (double)right_distortion[2], (double)right_distortion[3]);
            std::cout << "Right distortion coefficients loaded successfully" << std::endl;
        } else {
            std::cout << "Failed to load right distortion coefficients" << std::endl;
        }
        
        // Load extrinsics and compute relative transform
        cv::FileNode left_T_BS = camera["left_T_BS"];
        cv::FileNode right_T_BS = camera["right_T_BS"];
        
        std::cout << "Left T_BS size: " << left_T_BS.size() << std::endl;
        std::cout << "Right T_BS size: " << right_T_BS.size() << std::endl;
        
        if (!left_T_BS.empty() && !right_T_BS.empty() && 
            left_T_BS.size() == 16 && right_T_BS.size() == 16) {
            
            cv::Mat T_B_left = (cv::Mat_<double>(4, 4) << 
                (double)left_T_BS[0], (double)left_T_BS[1], (double)left_T_BS[2], (double)left_T_BS[3],
                (double)left_T_BS[4], (double)left_T_BS[5], (double)left_T_BS[6], (double)left_T_BS[7],
                (double)left_T_BS[8], (double)left_T_BS[9], (double)left_T_BS[10], (double)left_T_BS[11],
                (double)left_T_BS[12], (double)left_T_BS[13], (double)left_T_BS[14], (double)left_T_BS[15]);
                
            cv::Mat T_B_right = (cv::Mat_<double>(4, 4) << 
                (double)right_T_BS[0], (double)right_T_BS[1], (double)right_T_BS[2], (double)right_T_BS[3],
                (double)right_T_BS[4], (double)right_T_BS[5], (double)right_T_BS[6], (double)right_T_BS[7],
                (double)right_T_BS[8], (double)right_T_BS[9], (double)right_T_BS[10], (double)right_T_BS[11],
                (double)right_T_BS[12], (double)right_T_BS[13], (double)right_T_BS[14], (double)right_T_BS[15]);
            
            // Compute T_left_right = T_B_right.inv() * T_B_left
            // This gives us the transform from left camera to right camera coordinate system
            T_left_right = T_B_right.inv() * T_B_left;
            std::cout << "Camera extrinsics loaded and relative transform computed successfully" << std::endl;
        } else {
            std::cout << "Failed to load camera extrinsics" << std::endl;
        }
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
