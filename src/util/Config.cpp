#include "Config.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <spdlog/spdlog.h>

namespace lightweight_vio {

bool Config::load(const std::string& config_file) {
    std::cout << "[DEBUG] Config::load called with file: " << config_file << std::endl;
    cv::FileStorage fs(config_file, cv::FileStorage::READ);
    
    if (!fs.isOpened()) {
        std::cerr << "Failed to open config file: " << config_file << std::endl;
        return false;
    }
    
    std::cout << "[DEBUG] Config file opened successfully" << std::endl;
    
    // Feature Detection Parameters
    cv::FileNode feature_detection = fs["feature_detection"];
    if (!feature_detection.empty()) {
        m_max_features = (int)feature_detection["max_features"];
        m_quality_level = (double)feature_detection["quality_level"];
        m_min_distance = (double)feature_detection["min_distance"];
    }
    
    // Optical Flow Parameters
    cv::FileNode optical_flow = fs["optical_flow"];
    if (!optical_flow.empty()) {
        m_window_size = (int)optical_flow["window_size"];
        m_max_level = (int)optical_flow["max_level"];
        m_max_iterations = (int)optical_flow["max_iterations"];
        m_epsilon = (double)optical_flow["epsilon"];
        m_error_threshold = (double)optical_flow["error_threshold"];
        m_max_movement = (double)optical_flow["max_movement"];
        m_min_eigen_threshold = (double)optical_flow["min_eigen_threshold"];
    }
    
    // Outlier Rejection Parameters
    cv::FileNode outlier_rejection = fs["outlier_rejection"];
    if (!outlier_rejection.empty()) {
        m_fundamental_threshold = (double)outlier_rejection["fundamental_threshold"];
        m_fundamental_confidence = (double)outlier_rejection["fundamental_confidence"];
        m_max_movement_distance = (double)outlier_rejection["max_movement_distance"];
        m_max_velocity_change = (double)outlier_rejection["max_velocity_change"];
        m_min_points_for_ransac = (int)outlier_rejection["min_points_for_ransac"];
    }
    
    // Stereo Matching Parameters
    cv::FileNode stereo_matching = fs["stereo_matching"];
    if (!stereo_matching.empty()) {
        // Stereo optical flow parameters
        m_stereo_window_size = (int)stereo_matching["window_size"];
        m_stereo_max_level = (int)stereo_matching["max_level"];
        m_stereo_max_iterations = (int)stereo_matching["max_iterations"];
        m_stereo_min_eigen_threshold = (double)stereo_matching["min_eigen_threshold"];
        
        // Stereo matching thresholds
        m_stereo_error_threshold = (double)stereo_matching["error_threshold"];
        m_min_disparity = (double)stereo_matching["min_disparity"];
        m_max_disparity = (double)stereo_matching["max_disparity"];
        m_max_y_difference = (double)stereo_matching["max_y_difference"];
        m_epipolar_threshold = (double)stereo_matching["epipolar_threshold"];
    }
    
    // Global Depth Parameters (used throughout the VIO system)
    cv::FileNode depth = fs["depth"];
    if (!depth.empty()) {
        m_min_depth = (double)depth["min_depth"];
        m_max_depth = (double)depth["max_depth"];
    }
    
    // Triangulation Parameters (specific to 3D point generation)
    cv::FileNode triangulation = fs["triangulation"];
    if (!triangulation.empty()) {
        m_max_reprojection_error = (double)triangulation["max_reprojection_error"];
        m_min_parallax = (double)triangulation["min_parallax"];
    }
    
    // Keyframe Parameters
    cv::FileNode keyframe = fs["keyframe"];
    if (!keyframe.empty()) {
        m_keyframe_interval = (int)keyframe["interval"];
        spdlog::info("Loaded keyframe interval from config: {}", m_keyframe_interval);
    } else {
        spdlog::warn("Keyframe section not found in config, using default: {}", m_keyframe_interval);
    }
    
    // Pose Optimization Parameters
    cv::FileNode pose_optimization = fs["pose_optimization"];
    if (!pose_optimization.empty()) {
        if (pose_optimization["max_iterations"].isInt()) {
            m_pose_max_iterations = (int)pose_optimization["max_iterations"];
        }
        if (pose_optimization["function_tolerance"].isReal()) {
            m_pose_function_tolerance = (double)pose_optimization["function_tolerance"];
        }
        if (pose_optimization["gradient_tolerance"].isReal()) {
            m_pose_gradient_tolerance = (double)pose_optimization["gradient_tolerance"];
        }
        if (pose_optimization["parameter_tolerance"].isReal()) {
            m_pose_parameter_tolerance = (double)pose_optimization["parameter_tolerance"];
        }
        if (pose_optimization["use_robust_kernel"].isInt()) {
            m_use_robust_kernel = (bool)(int)pose_optimization["use_robust_kernel"];
        }
        if (pose_optimization["huber_delta_mono"].isReal()) {
            m_huber_delta_mono = (double)pose_optimization["huber_delta_mono"];
        }
        if (pose_optimization["huber_delta_stereo"].isReal()) {
            m_huber_delta_stereo = (double)pose_optimization["huber_delta_stereo"];
        }
        if (pose_optimization["enable_outlier_detection"].isInt()) {
            m_enable_outlier_detection = (bool)(int)pose_optimization["enable_outlier_detection"];
        }
        if (pose_optimization["outlier_detection_rounds"].isInt()) {
            m_outlier_detection_rounds = (int)pose_optimization["outlier_detection_rounds"];
        }
        if (pose_optimization["enable_solver_logging"].isInt()) {
            m_enable_pose_solver_logging = (bool)(int)pose_optimization["enable_solver_logging"];
        }
    }
    
    // Camera Parameters
    cv::FileNode camera = fs["camera"];
    if (!camera.empty()) {
        m_image_width = (int)camera["image_width"];
        m_image_height = (int)camera["image_height"];
        m_border_size = (int)camera["border_size"];
        
        // Load camera intrinsics
        cv::FileNode left_intrinsics = camera["left_intrinsics"];
        cv::FileNode right_intrinsics = camera["right_intrinsics"];
        cv::FileNode left_distortion = camera["left_distortion"];
        cv::FileNode right_distortion = camera["right_distortion"];
        
        if (!left_intrinsics.empty() && left_intrinsics.size() == 4) {
            m_left_camera_matrix = (cv::Mat_<double>(3, 3) << 
                (double)left_intrinsics[0], 0, (double)left_intrinsics[2],
                0, (double)left_intrinsics[1], (double)left_intrinsics[3],
                0, 0, 1);
        }
        
        if (!right_intrinsics.empty() && right_intrinsics.size() == 4) {
            m_right_camera_matrix = (cv::Mat_<double>(3, 3) << 
                (double)right_intrinsics[0], 0, (double)right_intrinsics[2],
                0, (double)right_intrinsics[1], (double)right_intrinsics[3],
                0, 0, 1);
        }
        
        if (!left_distortion.empty() && left_distortion.size() == 4) {
            m_left_dist_coeffs = (cv::Mat_<double>(1, 4) << 
                (double)left_distortion[0], (double)left_distortion[1],
                (double)left_distortion[2], (double)left_distortion[3]);
        }
        
        if (!right_distortion.empty() && right_distortion.size() == 4) {
            m_right_dist_coeffs = (cv::Mat_<double>(1, 4) << 
                (double)right_distortion[0], (double)right_distortion[1],
                (double)right_distortion[2], (double)right_distortion[3]);
        }
        
        // Load extrinsics and compute relative transform
        cv::FileNode left_T_BS = camera["left_T_BS"];
        cv::FileNode right_T_BS = camera["right_T_BS"];
        
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
            m_T_left_right = T_B_right.inv() * T_B_left;
        }
    }
    
    // Performance Parameters
    cv::FileNode performance = fs["performance"];
    if (!performance.empty()) {
        m_enable_timing = (bool)(int)performance["enable_timing"];
        m_enable_debug_output = (bool)(int)performance["enable_debug_output"];
    }
    
    fs.release();
    return true;
}

} // namespace lightweight_vio
