#include "PoseOptimizer.h"
#include "../database/Frame.h"
#include "../database/MapPoint.h"

namespace lightweight_vio
{

    PoseOptimizer::PoseOptimizer(const Config &config)
        : m_config(config), m_se3_manifold(std::make_unique<factor::SE3Manifold>())
    {
    }

    OptimizationResult PoseOptimizer::optimize_pose(std::shared_ptr<Frame> frame)
    {
        OptimizationResult result;

        std::cout << "[DEBUG] Starting pose optimization..." << std::endl;

        // Create Ceres problem
        ceres::Problem problem;

        // Convert frame pose to SE3 tangent space
        Eigen::Vector6d pose_params = frame_to_se3_tangent(frame);
        std::cout << "[DEBUG] Pose params: [" << pose_params.transpose() << "]" << std::endl;

        // Add parameter block first
        problem.AddParameterBlock(pose_params.data(), 6);
        std::cout << "[DEBUG] Parameter block added" << std::endl;

        // Set SE3 manifold for pose parameterization
        problem.SetManifold(pose_params.data(), m_se3_manifold.get());
        std::cout << "[DEBUG] SE3 manifold set" << std::endl;

        // Get camera parameters from frame
        double fx, fy, cx, cy;
        frame->get_camera_intrinsics(fx, fy, cx, cy);
        std::cout << "[DEBUG] Camera intrinsics: fx=" << fx << ", fy=" << fy << ", cx=" << cx << ", cy=" << cy << std::endl;
        factor::CameraParameters camera_params(fx, fy, cx, cy);

        // Add observations to the problem
        std::vector<ObservationInfo> observations;
        std::vector<int> feature_indices; // Track which features correspond to observations
        int num_valid_observations = 0;

        std::cout << "[DEBUG] Adding observations to problem..." << std::endl;

        // Outlier flags will be initialized when features are undistorted

        // Add mono PnP observations from frame's map points
        const auto &map_points = frame->get_map_points();
        std::cout << "[DEBUG] Frame has " << map_points.size() << " map points" << std::endl;
        for (size_t i = 0; i < map_points.size(); ++i)
        {
            auto mp = map_points[i];
            if (!mp || mp->is_bad())
            {
                continue;
            }

            // Get 3D world point
            Eigen::Vector3d world_point = mp->get_position().cast<double>();

            // Get 2D observation from feature
            if (i >= frame->get_features().size())
            {
                continue;

                // Get camera parameters from frame
                double fx, fy, cx, cy;
                frame->get_camera_intrinsics(fx, fy, cx, cy);
                factor::CameraParameters camera_params(fx, fy, cx, cy);

                // Add observations to the problem
                std::vector<ObservationInfo> observations;
                std::vector<int> feature_indices; // Track which features correspond to observations
                int num_valid_observations = 0;

                // Outlier flags will be initialized when features are undistorted

                // Add mono PnP observations from frame's map points
                const auto &map_points = frame->get_map_points();
                for (size_t i = 0; i < map_points.size(); ++i)
                {
                    auto mp = map_points[i];
                    if (!mp || mp->is_bad())
                    {
                        continue;
                    }

                    // Get 3D world point
                    Eigen::Vector3d world_point = mp->get_position().cast<double>();

                    // Get 2D observation from feature
                    if (i >= frame->get_features().size())
                    {
                        continue;
                    }

                    auto feature = frame->get_features()[i];

                    // Undistort the feature point
                    cv::Point2f distorted_point(feature->get_u(), feature->get_v());
                    cv::Point2f undistorted_point = frame->undistort_point(distorted_point);
                    Eigen::Vector2d observation(undistorted_point.x, undistorted_point.y);

                    // Add mono PnP observation
                    auto obs_info = add_mono_observation(
                        problem, pose_params.data(), world_point, observation, camera_params);

                    if (obs_info.residual_id)
                    {
                        observations.push_back(obs_info);
                        feature_indices.push_back(i);
                        num_valid_observations++;
                    }
                }

                // Check if we have enough observations
                std::cout << "[DEBUG] Total valid observations: " << num_valid_observations << std::endl;
                if (num_valid_observations < 5)
                {
                    std::cout << "[DEBUG] Not enough observations for optimization" << std::endl;
                    result.success = false;
                    result.num_inliers = 0;
                    return result;
                }

                // Setup solver options
                ceres::Solver::Options options = setup_solver_options();

                // Perform outlier detection rounds if enabled (ORB-SLAM3 style)
                if (m_config.enable_outlier_detection)
                {
                    for (int round = 0; round < m_config.outlier_detection_rounds; ++round)
                    {
                        // Solve
                        ceres::Solver::Summary summary;
                        ceres::Solve(options, &problem, &summary);

                        // Detect outliers and update frame's outlier flags
                        double *pose_data = pose_params.data();
                        detect_outliers(const_cast<double const *const *>(&pose_data),
                                        observations, feature_indices, frame);

                        // Remove outlier residual blocks for next iteration
                        if (round < m_config.outlier_detection_rounds - 1)
                        {
                            // Disable outlier residuals by setting them to level 1
                            for (size_t i = 0; i < observations.size(); ++i)
                            {
                                int feature_idx = feature_indices[i];
                                if (frame->get_outlier_flag(feature_idx))
                                {
                                    // Note: Ceres doesn't have SetLevel, so we'll handle this differently
                                    // For now, we'll recreate the problem without outliers
                                }
                            }
                        }

                        // Update result
                        result.final_cost = summary.final_cost;
                        result.num_iterations += summary.iterations.size();
                        result.success = (summary.termination_type == ceres::CONVERGENCE);
                    }
                }
                else
                {
                    // Single solve without outlier detection
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);

                    result.success = (summary.termination_type == ceres::CONVERGENCE);
                    result.final_cost = summary.final_cost;
                    result.num_iterations = summary.iterations.size();
                }

                // Count final inliers/outliers
                result.num_inliers = 0;
                result.num_outliers = 0;
                const auto &outlier_flags = frame->get_outlier_flags();
                for (bool is_outlier : outlier_flags)
                {
                    if (is_outlier)
                    {
                        result.num_outliers++;
                    }
                    else
                    {
                        result.num_inliers++;
                    }
                }

                // Update result
                result.optimized_pose = se3_tangent_to_matrix(pose_params);

                // Update frame pose if optimization was successful
                if (result.success)
                {
                    frame->set_Twb(result.optimized_pose);
                }

                if (m_config.print_summary)
                {
                    std::cout << "Pose optimization: " << result.num_inliers << " inliers, "
                              << result.num_outliers << " outliers" << std::endl;
                }

                return result;
            }
        }
    }

    // Helper function implementations moved outside optimize_pose
    ObservationInfo PoseOptimizer::add_mono_observation(
        ceres::Problem &problem,
        double *pose_params,
        const Eigen::Vector3d &world_point,
        const Eigen::Vector2d &observation,
        const factor::CameraParameters &camera_params)
    {

        // Create information matrix for pixel observations
        // You can adjust pixel_noise based on feature detection accuracy
        Eigen::Matrix2d information = create_information_matrix(1.0); // 1 pixel std dev

        // Create mono PnP cost function with information matrix
        auto cost_function = new factor::MonoPnPFactor(observation, world_point, camera_params, information);

        // Create robust loss function if enabled
        ceres::LossFunction *loss_function = nullptr;
        if (m_config.use_robust_kernel)
        {
            loss_function = create_robust_loss(m_config.huber_delta_mono);
        }

        // Add residual block
        auto residual_id = problem.AddResidualBlock(
            cost_function, loss_function, pose_params);

        return ObservationInfo(residual_id, cost_function);
    }

    int PoseOptimizer::detect_outliers(double const *const *pose_params,
                                       const std::vector<ObservationInfo> &observations,
                                       const std::vector<int> &feature_indices,
                                       std::shared_ptr<Frame> frame)
    {
        int num_inliers = 0;

        // Chi-square threshold for 2DOF (95% confidence)
        const double chi2_threshold = 5.99;

        for (size_t i = 0; i < observations.size(); ++i)
        {
            // Use our custom Chi-square computation with information matrix
            double chi2_error = observations[i].cost_function->compute_chi_square(pose_params);

            // Mark as outlier if above threshold
            bool is_outlier = (chi2_error > chi2_threshold);
            int feature_idx = feature_indices[i];
            frame->set_outlier_flag(feature_idx, is_outlier);

            if (!is_outlier)
            {
                num_inliers++;
            }

            // Debug output
            if (m_config.print_summary && is_outlier)
            {
                std::cout << "Outlier detected: feature " << feature_idx
                          << ", chi2 = " << chi2_error << std::endl;
            }
        }

        return num_inliers;
    }

    ceres::Solver::Options PoseOptimizer::setup_solver_options() const
    {
        ceres::Solver::Options options;

        options.max_num_iterations = m_config.max_iterations;
        options.function_tolerance = m_config.function_tolerance;
        options.gradient_tolerance = m_config.gradient_tolerance;
        options.parameter_tolerance = m_config.parameter_tolerance;

        options.linear_solver_type = m_config.linear_solver_type;
        options.use_explicit_schur_complement = m_config.use_explicit_schur_complement;

        options.minimizer_progress_to_stdout = m_config.minimizer_progress_to_stdout;

        return options;
    }

    Eigen::Vector6d PoseOptimizer::frame_to_se3_tangent(std::shared_ptr<Frame> frame) const
    {
        // Get frame pose (Twb)
        Eigen::Matrix4f Twb = frame->get_Twb();

        // Convert to double precision
        Eigen::Matrix4d Twb_d = Twb.cast<double>();

        // Convert to Sophus SE3 and extract tangent space
        Sophus::SE3d se3(Twb_d);
        return se3.log();
    }

    Eigen::Matrix4f PoseOptimizer::se3_tangent_to_matrix(const Eigen::Vector6d &se3_tangent) const
    {
        // Convert tangent space to SE3
        Sophus::SE3d se3 = Sophus::SE3d::exp(se3_tangent);

        // Convert to 4x4 matrix and cast to float
        return se3.matrix().cast<float>();
    }

    ceres::LossFunction *PoseOptimizer::create_robust_loss(double delta) const
    {
        return new ceres::HuberLoss(delta);
    }

    Eigen::Matrix2d PoseOptimizer::create_information_matrix(double pixel_noise) const
    {
        // Information matrix is inverse of covariance matrix
        // For isotropic pixel noise: Covariance = sigma^2 * I
        // Information = (1/sigma^2) * I
        double precision = 1.0 / (pixel_noise * pixel_noise);
        return precision * Eigen::Matrix2d::Identity();
    }

} // namespace lightweight_vio
