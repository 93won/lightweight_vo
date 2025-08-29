#include <module/PoseOptimizer.h>
#include <database/Frame.h>
#include <database/MapPoint.h>
#include <factor/Parameters.h>
#include <spdlog/spdlog.h>
#include <sstream>
#include <algorithm>
#include <numeric>

namespace lightweight_vio
{

    PoseOptimizer::PoseOptimizer(const Config &config)
        : m_config(config)
    {
    }

    OptimizationResult PoseOptimizer::optimize_pose(std::shared_ptr<Frame> frame)
    {
        OptimizationResult result;

        // Create Ceres problem
        ceres::Problem problem;

        // Convert frame pose to SE3 tangent space
        Eigen::Vector6d pose_params = frame_to_se3_tangent(frame);

        // Add parameter block first
        problem.AddParameterBlock(pose_params.data(), 6);

        // Set SE3 local parameterization for pose parameterization
        auto se3_local_param = new factor::SE3LocalParameterization();
        problem.SetParameterization(pose_params.data(), se3_local_param);

        // Get camera parameters from frame
        double fx, fy, cx, cy;
        frame->get_camera_intrinsics(fx, fy, cx, cy);
        factor::CameraParameters camera_params(fx, fy, cx, cy);

        // Add observations to the problem
        std::vector<ObservationInfo> observations;
        std::vector<int> feature_indices; // Track which features correspond to observations
        int num_valid_observations = 0;

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

            // Add mono PnP observation with desired pixel noise standard deviation
            auto obs_info = add_mono_observation(problem, pose_params.data(), world_point, observation, camera_params, 2.0);

            // Debug: Check if projection makes sense for first few features
            if (num_valid_observations < 3) {
                spdlog::debug("[PROJECTION] Feature {}: pixel=({:.2f},{:.2f}), world=({:.2f},{:.2f},{:.2f})", 
                             i, observation.x(), observation.y(), 
                             world_point.x(), world_point.y(), world_point.z());
            }

            if (obs_info.residual_id)
            {
                observations.push_back(obs_info);
                feature_indices.push_back(i);
                num_valid_observations++;
            }
        }

        // Check if we have enough observations
        if (num_valid_observations < 5)
        {
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

                // Brief report
                spdlog::info("[CERES] Round {}: {}", round + 1, summary.BriefReport());

                // Detect outliers and update frame's outlier flags
                double *pose_data = pose_params.data();
                int num_inliers = detect_outliers(const_cast<double const *const *>(&pose_data), observations, feature_indices, frame);
                int num_outliers = observations.size() - num_inliers;
                
                spdlog::info("[OUTLIER] Round {}: {} inliers, {} outliers", 
                            round + 1, num_inliers, num_outliers);

                // Remove outlier residual blocks for next iteration
                if (round < m_config.outlier_detection_rounds - 1)
                {
                    // Outliers are already disabled via set_outlier() in detect_outliers()
                    // The cost functions will return zero residuals and jacobians for outliers
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

            // Brief report
            spdlog::info("[CERES] {}", summary.BriefReport());

            // Check outliers for final report even without outlier detection rounds
            double *pose_data = pose_params.data();
            int num_inliers = detect_outliers(const_cast<double const *const *>(&pose_data), observations, feature_indices, frame);
            int num_outliers = observations.size() - num_inliers;
            spdlog::info("[OUTLIER] Final: {} inliers, {} outliers", num_inliers, num_outliers);

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
            spdlog::info("[POSE] Optimization: {} inliers, {} outliers", 
                        result.num_inliers, result.num_outliers);
        }

        return result;
    }

    // Helper function implementations moved outside optimize_pose
    ObservationInfo PoseOptimizer::add_mono_observation(
        ceres::Problem &problem,
        double *pose_params,
        const Eigen::Vector3d &world_point,
        const Eigen::Vector2d &observation,
        const factor::CameraParameters &camera_params,
        double pixel_noise_std)
    {

        // Create information matrix for pixel observations
        Eigen::Matrix2d information = create_information_matrix(pixel_noise_std);

        // For now, assume camera is at body frame (identity transformation)
        // TODO: Get actual Tcb from frame configuration
        Eigen::Matrix4d Tcb = Eigen::Matrix4d::Identity();

        // Create mono PnP cost function with information matrix and Tcb
        auto cost_function = new factor::MonoPnPFactor(observation, world_point, camera_params, Tcb, information);

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

        // Chi-square threshold for 2DOF - use more relaxed threshold
        const double chi2_threshold = 5.991;  // Much more relaxed than 5.99

        // Collect chi2 values for statistics
        std::vector<double> inlier_chi2_values;
        std::vector<double> outlier_chi2_values;

        for (size_t i = 0; i < observations.size(); ++i)
        {
            // Use our custom Chi-square computation with information matrix
            double chi2_error = observations[i].cost_function->compute_chi_square(pose_params);

            // Mark as outlier if above threshold
            bool is_outlier = (chi2_error > chi2_threshold);
            int feature_idx = feature_indices[i];
            frame->set_outlier_flag(feature_idx, is_outlier);

            // Set outlier flag in the cost function to disable it for next optimization round
            observations[i].cost_function->set_outlier(is_outlier);

            if (!is_outlier)
            {
                num_inliers++;
                inlier_chi2_values.push_back(chi2_error);
            }
            else
            {
                outlier_chi2_values.push_back(chi2_error);
            }
        }

        // Print chi2 statistics
        if (!inlier_chi2_values.empty())
        {
            auto inlier_minmax = std::minmax_element(inlier_chi2_values.begin(), inlier_chi2_values.end());
            double inlier_sum = std::accumulate(inlier_chi2_values.begin(), inlier_chi2_values.end(), 0.0);
            double inlier_mean = inlier_sum / inlier_chi2_values.size();
            
            spdlog::info("[CHI2_STATS] Inliers ({}): min={:.3f}, max={:.3f}, mean={:.3f}", 
                        inlier_chi2_values.size(), *inlier_minmax.first, 
                        *inlier_minmax.second, inlier_mean);
        }

        if (!outlier_chi2_values.empty())
        {
            auto outlier_minmax = std::minmax_element(outlier_chi2_values.begin(), outlier_chi2_values.end());
            double outlier_sum = std::accumulate(outlier_chi2_values.begin(), outlier_chi2_values.end(), 0.0);
            double outlier_mean = outlier_sum / outlier_chi2_values.size();
            
            spdlog::info("[CHI2_STATS] Outliers ({}): min={:.3f}, max={:.3f}, mean={:.3f}", 
                        outlier_chi2_values.size(), *outlier_minmax.first, 
                        *outlier_minmax.second, outlier_mean);
        }

        // Debug: Print details for some outliers to understand what's wrong
        int debug_count = 0;
        Eigen::Map<const Eigen::Vector6d> se3_tangent(pose_params[0]);
        Sophus::SE3d current_pose = Sophus::SE3d::exp(se3_tangent);
        
        // Convert matrix to string for logging
        std::stringstream ss;
        ss << current_pose.matrix();
        spdlog::debug("[OUTLIER_DEBUG] Current pose Twb:\n{}", ss.str());
        
        for (size_t i = 0; i < observations.size() && debug_count < 3; ++i)
        {
            double chi2_error = observations[i].cost_function->compute_chi_square(pose_params);
            if (chi2_error > chi2_threshold)
            {
                int feature_idx = feature_indices[i];
                auto feature = frame->get_features()[feature_idx];
                auto mp = frame->get_map_points()[feature_idx];
                
                // Manually project to see what the expected pixel should be
                Eigen::Vector3d world_pos = mp->get_position().cast<double>();
                
                // Transform to camera coordinates: Pc = Rcw * Pw + tcw
                Eigen::Matrix3d Rwb = current_pose.rotationMatrix();
                Eigen::Vector3d twb = current_pose.translation();
                Eigen::Matrix3d Rbw = Rwb.transpose();
                Eigen::Vector3d tbw = -Rbw * twb;
                
                // Assuming Tcb = Identity (camera at body frame)
                Eigen::Vector3d point_camera = Rbw * world_pos + tbw;
                
                // Project to image plane
                double fx, fy, cx, cy;
                frame->get_camera_intrinsics(fx, fy, cx, cy);
                
                if (point_camera.z() > 0) {
                    double u_proj = fx * point_camera.x() / point_camera.z() + cx;
                    double v_proj = fy * point_camera.y() / point_camera.z() + cy;
                    
                    auto map_point = frame->get_map_point(feature_idx);
                    spdlog::debug("[OUTLIER_DEBUG] Feature {}: chi2={:.3f}, observed=({:.1f},{:.1f}), projected=({:.1f},{:.1f}), world=({:.2f},{:.2f},{:.2f})", 
                                 feature_idx, chi2_error, feature->get_u(), feature->get_v(),
                                 u_proj, v_proj, world_pos.x(), world_pos.y(), world_pos.z());
                } else {
                    spdlog::debug("[OUTLIER_DEBUG] Feature {}: chi2={:.3f}, BEHIND_CAMERA: z={:.2f}", 
                                 feature_idx, chi2_error, point_camera.z());
                }
                debug_count++;
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

        // Brief report only, no verbose stdout output
        options.minimizer_progress_to_stdout = false;

        return options;
    }

    Eigen::Vector6d PoseOptimizer::frame_to_se3_tangent(std::shared_ptr<Frame> frame) const
    {
        // Get frame pose (Twb)
        Eigen::Matrix4f Twb = frame->get_Twb();

        // Convert to double precision
        Eigen::Matrix4d Twb_d = Twb.cast<double>();

        // Ensure rotation matrix orthogonality using SVD
        Eigen::Matrix3d R = Twb_d.block<3, 3>(0, 0);
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        R = svd.matrixU() * svd.matrixV().transpose();
        
        // Ensure proper rotation (det(R) = 1)
        if (R.determinant() < 0) {
            R = -R;
        }
        
        // Reconstruct the pose matrix with orthogonalized rotation
        Twb_d.block<3, 3>(0, 0) = R;

        // Convert to Sophus SE3 and extract tangent space
        Sophus::SE3d se3(Twb_d);
        return se3.log();
    }

    Eigen::Matrix4f PoseOptimizer::se3_tangent_to_matrix(const Eigen::Vector6d &se3_tangent) const
    {
        // Convert tangent space to SE3
        Sophus::SE3d se3 = Sophus::SE3d::exp(se3_tangent);

        // Get the transformation matrix
        Eigen::Matrix4d pose_matrix = se3.matrix();
        
        // Ensure rotation matrix orthogonality using SVD
        Eigen::Matrix3d R = pose_matrix.block<3, 3>(0, 0);
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        R = svd.matrixU() * svd.matrixV().transpose();
        
        // Ensure proper rotation (det(R) = 1)
        if (R.determinant() < 0) {
            R = -R;
        }
        
        // Reconstruct the pose matrix with orthogonalized rotation
        pose_matrix.block<3, 3>(0, 0) = R;

        // Convert to 4x4 matrix and cast to float
        return pose_matrix.cast<float>();
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
