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
        int m_max_observation_without_mappoint = 5;  // Remove features after this many observations without map point

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

        // Pose Optimization Parameters
        int m_pose_max_iterations = 10;
        double m_pose_function_tolerance = 1e-6;
        double m_pose_gradient_tolerance = 1e-10;
        double m_pose_parameter_tolerance = 1e-8;
        bool m_enable_pose_solver_logging = false;

        // Robust kernel parameters
        bool m_use_robust_kernel = true;
        double m_huber_delta_mono = 5.99;   // sqrt(chi2_2dof_95%)
        double m_huber_delta_stereo = 7.81; // sqrt(chi2_3dof_95%)

        // Outlier detection
        bool m_enable_outlier_detection = true;
        int m_outlier_detection_rounds = 3;

        // Logging
        bool m_minimizer_progress_to_stdout = false;
        bool m_print_summary = false;

        // Camera Parameters
        int m_image_width = 752;
        int m_image_height = 480;
        int m_border_size = 1; // Baseline for convenience (EuRoC dataset)
        double m_baseline = 0.11;

        // Performance Parameters
        bool m_enable_timing = true;
        bool m_enable_debug_output = true;

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