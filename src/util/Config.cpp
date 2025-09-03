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
        
        // Grid-based feature distribution parameters
        if (!feature_detection["grid_cols"].empty()) {
            m_grid_cols = (int)feature_detection["grid_cols"];
        }
        if (!feature_detection["grid_rows"].empty()) {
            m_grid_rows = (int)feature_detection["grid_rows"];
        }
        if (!feature_detection["max_features_per_grid"].empty()) {
            m_max_features_per_grid = (int)feature_detection["max_features_per_grid"];
        }
        if (!feature_detection["max_observation_without_mappoint"].empty()) {
            m_max_observation_without_mappoint = (int)feature_detection["max_observation_without_mappoint"];
        }
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
        m_F_threshold = (double)optical_flow["F_threshold"];
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
    cv::FileNode keyframe_mgmt = fs["keyframe_management"];
    if (!keyframe_mgmt.empty()) {
        m_grid_coverage_ratio = (double)keyframe_mgmt["grid_coverage_ratio"];
        m_keyframe_window_size = (int)keyframe_mgmt["keyframe_window_size"];
        spdlog::info("Loaded keyframe management from config: grid_coverage_ratio={}, window_size={}", 
                    m_grid_coverage_ratio, m_keyframe_window_size);
    } else {
        spdlog::warn("Keyframe management section not found in config, using defaults: grid_coverage_ratio={}, window_size={}", 
                    m_grid_coverage_ratio, m_keyframe_window_size);
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
        if (pose_optimization["minimizer_progress_to_stdout"].isInt()) {
            m_minimizer_progress_to_stdout = (bool)(int)pose_optimization["minimizer_progress_to_stdout"];
        }
        if (pose_optimization["print_summary"].isInt()) {
            m_print_summary = (bool)(int)pose_optimization["print_summary"];
        }
    }
    
    // Camera Parameters
    cv::FileNode camera = fs["camera"];
    if (!camera.empty()) {
        m_image_width = (int)camera["image_width"];
        m_image_height = (int)camera["image_height"];
        m_border_size = (int)camera["border_size"];
        
        // Load baseline if specified
        if (!camera["baseline"].empty()) {
            m_baseline = (double)camera["baseline"];
        }
        
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
        
        // Load extrinsics (T_BC - camera to body transform)
        cv::FileNode left_T_BC = camera["left_T_BC"];
        cv::FileNode right_T_BC = camera["right_T_BC"];
        
        if (!left_T_BC.empty() && !right_T_BC.empty() && 
            left_T_BC.size() == 16 && right_T_BC.size() == 16) {
            
            cv::Mat T_BC_left = (cv::Mat_<double>(4, 4) << 
                (double)left_T_BC[0], (double)left_T_BC[1], (double)left_T_BC[2], (double)left_T_BC[3],
                (double)left_T_BC[4], (double)left_T_BC[5], (double)left_T_BC[6], (double)left_T_BC[7],
                (double)left_T_BC[8], (double)left_T_BC[9], (double)left_T_BC[10], (double)left_T_BC[11],
                (double)left_T_BC[12], (double)left_T_BC[13], (double)left_T_BC[14], (double)left_T_BC[15]);
                
            cv::Mat T_BC_right = (cv::Mat_<double>(4, 4) << 
                (double)right_T_BC[0], (double)right_T_BC[1], (double)right_T_BC[2], (double)right_T_BC[3],
                (double)right_T_BC[4], (double)right_T_BC[5], (double)right_T_BC[6], (double)right_T_BC[7],
                (double)right_T_BC[8], (double)right_T_BC[9], (double)right_T_BC[10], (double)right_T_BC[11],
                (double)right_T_BC[12], (double)right_T_BC[13], (double)right_T_BC[14], (double)right_T_BC[15]);
            
            // Store individual T_BC matrices (camera to body transforms)
            m_T_left_BC = T_BC_left.clone();
            m_T_right_BC = T_BC_right.clone();
            
            // Compute stereo baseline transform: T_left_right = T_BC_right.inv() * T_BC_left
            // This gives left-to-right camera transformation for stereo triangulation
            m_T_left_right = T_BC_right.inv() * T_BC_left;
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
