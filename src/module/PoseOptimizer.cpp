#include <module/PoseOptimizer.h>
#include <database/Frame.h>
#include <database/MapPoint.h>
#include <factor/Parameters.h>
#include <util/Config.h>
#include <spdlog/spdlog.h>
#include <sstream>
#include <algorithm>
#include <numeric>

namespace lightweight_vio
{

    PoseOptimizer::PoseOptimizer()
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

        // Set SE3 global parameterization for pose parameterization
        auto se3_global_param = new factor::SE3GlobalParameterization();
        problem.SetParameterization(pose_params.data(), se3_global_param);

        // Get camera parameters from frame
        double fx, fy, cx, cy;
        frame->get_camera_intrinsics(fx, fy, cx, cy);
        factor::CameraParameters camera_params(fx, fy, cx, cy);
        
        // DEBUG: Print camera intrinsics
        spdlog::debug("[DEBUG_CAM] Camera intrinsics: fx={:.2f}, fy={:.2f}, cx={:.2f}, cy={:.2f}",
                     fx, fy, cx, cy);

        // Add observations to the problem
        std::vector<ObservationInfo> observations;
        std::vector<int> feature_indices; // Track which features correspond to observations
        int num_valid_observations = 0;
        int num_excluded_outliers = 0;

        // Add mono PnP observations from frame's map points
        const auto &map_points = frame->get_map_points();
        
        for (size_t i = 0; i < map_points.size(); ++i)
        {
            auto mp = map_points[i];
            if (!mp || mp->is_bad())
            {
                continue;
            }

            // Give outliers a second chance - don't exclude them immediately
            // Only exclude if they've been consistently outliers for multiple frames
            // For now, let all features with map points participate in optimization
            bool is_previous_outlier = frame->get_outlier_flag(i);
            if (is_previous_outlier) {
                // Still include but track that it was an outlier
                num_excluded_outliers++; // This now means "previous outliers given another chance"
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
            auto obs_info = add_mono_observation(problem, pose_params.data(), world_point, observation, camera_params, frame, 2.0);

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

        // Get global config
        const auto& config = Config::getInstance();

        // Perform outlier detection rounds if enabled
        if (config.m_enable_outlier_detection)
        {
            double initial_cost = 0.0;
            double final_cost = 0.0;
            int total_iterations = 0;
            
            // Store initial pose parameters for resetting each round (ORB-SLAM style)
            Eigen::Vector6d initial_pose_params = pose_params;
            
            for (int round = 0; round < config.m_outlier_detection_rounds; ++round)
            {
                // Reset pose to initial value for each round
                if (round > 0) {
                    pose_params = initial_pose_params;
                    spdlog::debug("[POSE_OPT] Round {}: Reset pose to initial value", round);
                }
                
                // Solve
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);

                // Store costs for summary
                if (round == 0) {
                    initial_cost = summary.initial_cost;
                }
                final_cost = summary.final_cost;
                total_iterations += summary.iterations.size();

                // Detect outliers and update frame's outlier flags
                double *pose_data = pose_params.data();
                int num_inliers = detect_outliers(const_cast<double const *const *>(&pose_data), observations, feature_indices, frame);
                int num_outliers = observations.size() - num_inliers;

                // Remove outlier residual blocks for next iteration
                if (round < config.m_outlier_detection_rounds - 1)
                {
                    // Outliers are already disabled via set_outlier() in detect_outliers()
                    // The cost functions will return zero residuals and jacobians for outliers
                }

                // Update result
                result.final_cost = summary.final_cost;
                result.num_iterations += summary.iterations.size();
                result.success = (summary.termination_type == ceres::CONVERGENCE);
            }
            
            // Print consolidated optimization summary
            double *pose_data = pose_params.data();
            int final_inliers = detect_outliers(const_cast<double const *const *>(&pose_data), observations, feature_indices, frame);
            int final_outliers = observations.size() - final_inliers;
            
            spdlog::info("[POSE_OPT] {} rounds: cost {:.3e} -> {:.3e}, {} iters, {} inliers/{} outliers", 
                        config.m_outlier_detection_rounds, initial_cost, final_cost, 
                        total_iterations, final_inliers, final_outliers);
        }
        else
        {
            // Single solve without outlier detection
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            // Check outliers for final report even without outlier detection rounds
            double *pose_data = pose_params.data();
            int num_inliers = detect_outliers(const_cast<double const *const *>(&pose_data), observations, feature_indices, frame);
            int num_outliers = observations.size() - num_inliers;
            
            spdlog::info("[POSE_OPT] Single solve: cost {:.3e} -> {:.3e}, {} iters, {} inliers/{} outliers", 
                        summary.initial_cost, summary.final_cost, summary.iterations.size(),
                        num_inliers, num_outliers);

            result.success = (summary.termination_type == ceres::CONVERGENCE);
            result.final_cost = summary.final_cost;
            result.num_iterations = summary.iterations.size();
        }

        // Count final inliers/outliers and disconnect outlier map points
        result.num_inliers = 0;
        result.num_outliers = 0;
        int disconnected_map_points = 0;
        
        const auto &outlier_flags = frame->get_outlier_flags();
        for (size_t i = 0; i < outlier_flags.size(); ++i)
        {
            bool is_outlier = outlier_flags[i];
            if (is_outlier)
            {
                result.num_outliers++;
                
                // Disconnect outlier feature from its map point
                auto map_point = frame->get_map_point(i);
                if (map_point && !map_point->is_bad()) {
                    // Remove observation from map point
                    map_point->remove_observation(frame);
                    
                    // Remove map point from frame
                    frame->set_map_point(i, nullptr);
                    
                    disconnected_map_points++;
                }
            }
            else
            {
                result.num_inliers++;
            }
        }
        
        if (disconnected_map_points > 0) {
            spdlog::warn("[POSE_OPT] Disconnected {} outlier map points", disconnected_map_points);
        }

        // Update result
        result.optimized_pose = se3_tangent_to_matrix(pose_params);


        // Update frame pose if optimization was successful
        if (result.success)
        {
            frame->set_Twb(result.optimized_pose);
        }

        // Summary is already printed in the optimization loop above
        // if (config.m_print_summary)
        // {
        //     spdlog::info("[POSE] Optimization: {} inliers, {} outliers", 
        //                 result.num_inliers, result.num_outliers);
        // }

        return result;
    }

    // Helper function implementations moved outside optimize_pose
    ObservationInfo PoseOptimizer::add_mono_observation(
        ceres::Problem &problem,
        double *pose_params,
        const Eigen::Vector3d &world_point,
        const Eigen::Vector2d &observation,
        const factor::CameraParameters &camera_params,
        std::shared_ptr<Frame> frame,
        double pixel_noise_std)
    {

        // Create information matrix for pixel observations
        Eigen::Matrix2d information = create_information_matrix(pixel_noise_std);

        // Get T_cb (body-to-camera transform) from frame directly - NO CONFIG ACCESS!
        const Eigen::Matrix4d& T_cb = frame->get_T_CB();
        
        // DEBUG: Print T_cb matrix from frame
        spdlog::debug("[DEBUG_FRAME_TCB] Using T_cb from Frame (cached):");
        for (int row = 0; row < 4; ++row) {
            spdlog::debug("[DEBUG_FRAME_TCB] [{:.6f}, {:.6f}, {:.6f}, {:.6f}]",
                         T_cb(row, 0), T_cb(row, 1), T_cb(row, 2), T_cb(row, 3));
        }

        // Create mono PnP cost function with information matrix and T_cb
        auto cost_function = new factor::MonoPnPFactor(observation, world_point, camera_params, T_cb, information);

        // Create robust loss function if enabled
        const Config& config = Config::getInstance();
        ceres::LossFunction *loss_function = nullptr;
        if (config.m_use_robust_kernel)
        {
            loss_function = create_robust_loss(config.m_huber_delta_mono);
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
            
            // Get previous outlier status
            bool was_outlier = frame->get_outlier_flag(feature_idx);
            
            // Update outlier flag - can be both set and cleared based on current chi2 test
            frame->set_outlier_flag(feature_idx, is_outlier);

            // Set outlier flag in the cost function to disable it for next optimization round
            observations[i].cost_function->set_outlier(is_outlier);

            if (!is_outlier)
            {
                num_inliers++;
                inlier_chi2_values.push_back(chi2_error);
                
                // Log recovery if this feature was previously an outlier
                if (was_outlier) {
                    spdlog::debug("[POSE_OPT] Feature {} recovered from outlier (chi2: {:.3f})", 
                                 feature_idx, chi2_error);
                }
            }
            else
            {
                outlier_chi2_values.push_back(chi2_error);
                
                // Log new outlier detection
                if (!was_outlier) {
                    spdlog::debug("[POSE_OPT] Feature {} marked as outlier (chi2: {:.3f})", 
                                 feature_idx, chi2_error);
                }
            }
        }

        // Print chi2 statistics (commented out to reduce log verbosity)
        // if (!inlier_chi2_values.empty())
        // {
        //     auto inlier_minmax = std::minmax_element(inlier_chi2_values.begin(), inlier_chi2_values.end());
        //     double inlier_sum = std::accumulate(inlier_chi2_values.begin(), inlier_chi2_values.end(), 0.0);
        //     double inlier_mean = inlier_sum / inlier_chi2_values.size();
        //     
        //     spdlog::info("[CHI2_STATS] Inliers ({}): min={:.3f}, max={:.3f}, mean={:.3f}", 
        //                 inlier_chi2_values.size(), *inlier_minmax.first, 
        //                 *inlier_minmax.second, inlier_mean);
        // }

        // if (!outlier_chi2_values.empty())
        // {
        //     auto outlier_minmax = std::minmax_element(outlier_chi2_values.begin(), outlier_chi2_values.end());
        //     double outlier_sum = std::accumulate(outlier_chi2_values.begin(), outlier_chi2_values.end(), 0.0);
        //     double outlier_mean = outlier_sum / outlier_chi2_values.size();
        //     
        //     spdlog::info("[CHI2_STATS] Outliers ({}): min={:.3f}, max={:.3f}, mean={:.3f}", 
        //                 outlier_chi2_values.size(), *outlier_minmax.first, 
        //                 *outlier_minmax.second, outlier_mean);
        // }

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
                Eigen::Vector3d t_wb = current_pose.translation();
                Eigen::Matrix3d Rbw = Rwb.transpose();
                Eigen::Vector3d t_bw = -Rbw * t_wb;
                
                // Get T_cb (body-to-camera transform) from configuration
                // T_cb = T_bc.inverse() where T_bc is camera-to-body transform
                const auto& config = Config::getInstance();
                cv::Mat T_bc_cv = config.left_T_BC();
                Eigen::Matrix4d T_bc;
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        T_bc(i, j) = T_bc_cv.at<double>(i, j);
                    }
                }
                Eigen::Matrix4d T_cb = T_bc.inverse();
                
                // Transform to camera coordinates: Pc = T_cb * (Rbw * Pw + t_bw)
                Eigen::Vector3d point_body = Rbw * world_pos + t_bw;
                Eigen::Vector4d point_body_h(point_body.x(), point_body.y(), point_body.z(), 1.0);
                Eigen::Vector4d point_camera_h = T_cb * point_body_h;
                Eigen::Vector3d point_camera = point_camera_h.head<3>();
                
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
        const auto& config = Config::getInstance();

        options.max_num_iterations = config.m_pose_max_iterations;
        options.function_tolerance = config.m_pose_function_tolerance;
        options.gradient_tolerance = config.m_pose_gradient_tolerance;
        options.parameter_tolerance = config.m_pose_parameter_tolerance;

        // Use fixed solver configuration for now
        options.linear_solver_type = ceres::DENSE_QR;
        options.use_explicit_schur_complement = false;

        // Logging configuration
        options.logging_type = config.m_enable_pose_solver_logging ? ceres::PER_MINIMIZER_ITERATION : ceres::SILENT;
        options.minimizer_progress_to_stdout = config.m_enable_pose_solver_logging;

        return options;
    }

    Eigen::Vector6d PoseOptimizer::frame_to_se3_tangent(std::shared_ptr<Frame> frame) const
    {
        // Get frame pose (T_wb)
        Eigen::Matrix4f T_wb = frame->get_Twb();

        // Convert to double precision
        Eigen::Matrix4d T_wb_d = T_wb.cast<double>();

        // Fix numerical precision issues from float->double conversion
        Eigen::Matrix3d R = T_wb_d.block<3, 3>(0, 0);
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        R = svd.matrixU() * svd.matrixV().transpose();
        
        // Ensure proper rotation (det(R) = 1)
        if (R.determinant() < 0) {
            R = -R;
        }
        
        // Reconstruct the pose matrix with orthogonalized rotation
        T_wb_d.block<3, 3>(0, 0) = R;

        // Now Sophus SE3 constructor will be happy
        Sophus::SE3d se3(T_wb_d);
        return se3.log();
    }

    Eigen::Matrix4f PoseOptimizer::se3_tangent_to_matrix(const Eigen::Vector6d &se3_tangent) const
    {
        // Convert tangent space to SE3 using Sophus (already guarantees proper SE3)
        Sophus::SE3d se3 = Sophus::SE3d::exp(se3_tangent);

        // Sophus already ensures proper SE3 structure, no need for SVD
        // Just convert to float at the end
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
