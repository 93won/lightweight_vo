/**
 * @file      Config.h
 * @brief     Defines a singleton class for managing configuration parameters.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-08-11
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace lightweight_vio
{

    class Config
    {
    public:
        static Config &getInstance()
        {
            static Config instance;
            return instance;
        }

        bool load(const std::string &config_file);

        // Functions for complex types only
        cv::TermCriteria term_criteria() const
        {
            return cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                    m_max_iterations, m_epsilon);
        }
        cv::TermCriteria stereo_term_criteria() const
        {
            return cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                    m_stereo_max_iterations, m_epsilon);
        }

        cv::Mat left_camera_matrix() const { return m_left_camera_matrix.clone(); }
        cv::Mat right_camera_matrix() const { return m_right_camera_matrix.clone(); }
        cv::Mat left_dist_coeffs() const { return m_left_dist_coeffs.clone(); }
        cv::Mat right_dist_coeffs() const { return m_right_dist_coeffs.clone(); }
        cv::Mat left_to_right_transform() const { return m_T_left_right.clone(); }
        cv::Mat left_T_BC() const { return m_T_left_BC.clone(); }
        cv::Mat right_T_BC() const { return m_T_right_BC.clone(); }

        // Public member variables for simple access

        // Feature Detection Parameters
        int m_max_features = 150;
        double m_quality_level = 0.01;
        double m_min_distance = 30.0;

        // Grid-based Feature Distribution Parameters
        int m_grid_cols = 20;
        int m_grid_rows = 10;
        int m_max_features_per_grid = 4;

        // Optical Flow Parameters
        int m_window_size = 21;
        int m_max_level = 3;
        int m_max_iterations = 30;
        double m_epsilon = 0.01;
        double m_error_threshold = 30.0;
        double m_max_movement = 100.0;
        double m_min_eigen_threshold = 5e-2;
        double m_F_threshold = 1.0;  // Fundamental matrix RANSAC distance threshold (pixels)

        // Stereo Matching Parameters
        int m_stereo_window_size = 31;
        int m_stereo_max_level = 4;
        int m_stereo_max_iterations = 50;
        double m_stereo_min_eigen_threshold = 1e-3;

        double m_stereo_error_threshold = 50.0;
        double m_min_disparity = 0.1;
        double m_max_disparity = 300.0;
        double m_max_y_difference = 20.0;
        double m_epipolar_threshold = 5.0;

        // Stereo Rectification Parameters
        double m_max_rectified_y_difference = 2.0;

        // Global Depth Parameters
        double m_min_depth = 0.1;
        double m_max_depth = 100.0;

        // Triangulation Parameters
        double m_max_reprojection_error = 10.0;
        double m_min_parallax = 0.5;

        // Keyframe Parameters
        double m_grid_coverage_ratio = 0.7;  // Add keyframe when coverage drops to this ratio of last keyframe's coverage
        int m_keyframe_window_size = 10;     // Number of keyframes to keep in sliding window
        double m_keyframe_time_threshold = 0.5;  // Force keyframe creation if time since last keyframe exceeds this (seconds)

        // PnP Optimization Parameters
        int m_pnp_max_iterations = 10;
        double m_pnp_function_tolerance = 1e-6;
        double m_pnp_gradient_tolerance = 1e-10;
        double m_pnp_parameter_tolerance = 1e-8;
        bool m_pnp_use_robust_kernel = true;
        // m_pnp_huber_delta_mono removed - hardcoded to 5.991
        bool m_pnp_enable_outlier_detection = true;
        int m_pnp_outlier_detection_rounds = 3;
        double m_pnp_max_observation_weight = 3.0;
        std::string m_pnp_information_matrix_mode = "observation_count"; // "observation_count", "reprojection_error"

        // Sliding Window Optimization Parameters  
        int m_sw_max_iterations = 20;
        double m_sw_function_tolerance = 1e-6;
        double m_sw_gradient_tolerance = 1e-10;
        double m_sw_parameter_tolerance = 1e-8;
        bool m_sw_use_robust_kernel = true;
        // m_sw_huber_delta removed - hardcoded to 5.991
        double m_sw_max_observation_weight = 3.0;

        // Legacy parameters (for backward compatibility)
        int m_pose_max_iterations = 10;
        double m_pose_function_tolerance = 1e-6;
        double m_pose_gradient_tolerance = 1e-10;
        double m_pose_parameter_tolerance = 1e-8;
        bool m_use_robust_kernel = true;
        // m_huber_delta_mono removed - hardcoded to 5.991
        // m_huber_delta_stereo removed - not used
        bool m_enable_outlier_detection = true;
        int m_outlier_detection_rounds = 3;
        double m_max_observation_weight = 3.0;

        // Camera Parameters
        int m_image_width = 752;
        int m_image_height = 480;
        int m_border_size = 1;

        // Performance Parameters
        bool m_enable_timing = true;
        bool m_enable_debug_output = false;

        // System Mode Parameters
        std::string m_system_mode = "VIO";  // "VO" or "VIO"
        
        // Gravity Estimation Parameters (VIO mode only)
        bool m_gravity_estimation_enable = true;
        int m_gravity_min_frames_for_estimation = 10;
        double m_gravity_magnitude = 9.81;

        // IMU Noise Model Parameters
        double m_gyro_noise_density = 1.6968e-04;      // rad/s/√Hz (gyro white noise)
        double m_gyro_random_walk = 1.9393e-05;        // rad/s²/√Hz (gyro bias diffusion)
        double m_accel_noise_density = 2.0000e-3;      // m/s²/√Hz (accel white noise)
        double m_accel_random_walk = 3.0000e-3;        // m/s³/√Hz (accel bias diffusion)

    private:
        Config() = default;

        // Private camera matrices (accessed through functions)
        cv::Mat m_left_camera_matrix;
        cv::Mat m_right_camera_matrix;
        cv::Mat m_left_dist_coeffs;
        cv::Mat m_right_dist_coeffs;
        cv::Mat m_T_left_right;
        cv::Mat m_T_left_BC;
        cv::Mat m_T_right_BC;
    };

} // namespace lightweight_vio