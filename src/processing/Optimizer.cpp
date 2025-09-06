#include <processing/Optimizer.h>
#include <database/Frame.h>
#include <database/MapPoint.h>
#include <optimization/Parameters.h>
#include <util/Config.h>
#include <spdlog/spdlog.h>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <set>
#include <unordered_map>
#include <thread>

namespace lightweight_vio
{

    // Define global mutexes for thread-safe access
    std::mutex PnPOptimizer::s_mappoint_mutex;
    std::mutex PnPOptimizer::s_keyframe_mutex;

    PnPOptimizer::PnPOptimizer()
    {
    }

    OptimizationResult PnPOptimizer::optimize_pose(std::shared_ptr<Frame> frame)
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
        
        // // DEBUG: Print camera intrinsics
        // spdlog::debug("[DEBUG_CAM] Camera intrinsics: fx={:.2f}, fy={:.2f}, cx={:.2f}, cy={:.2f}",
        //              fx, fy, cx, cy);

                // Add observations to the problem
        std::vector<ObservationInfo> observations;
        std::vector<int> feature_indices; // Track which features correspond to observations
        int num_valid_observations = 0;
        int num_excluded_outliers = 0;

        // Add mono PnP observations from frame's map points
        // Protect MapPoint access with mutex
        {
            std::lock_guard<std::mutex> lock(s_mappoint_mutex);
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

                // Get undistorted normalized coordinates and convert to pixel coordinates (consistent with CREATE_MP)
                cv::Point2f undistorted_pixel = feature->get_undistorted_coord();

                double undist_u = undistorted_pixel.x;
                double undist_v = undistorted_pixel.y;
                Eigen::Vector2d observation(undist_u, undist_v);

                // Add mono PnP observation with observation-based weighting
                int num_observations = mp->get_observation_count();
                auto obs_info = add_mono_observation(problem, pose_params.data(), world_point, observation, camera_params, frame, 2.0, num_observations);

                // Debug: Check if projection makes sense for first few features
                if (num_valid_observations < 3) {
                    // spdlog::debug("[PROJECTION] Feature {}: pixel=({:.2f},{:.2f}), world=({:.2f},{:.2f},{:.2f})", 
                    //              i, observation.x(), observation.y(), 
                    //              world_point.x(), world_point.y(), world_point.z());
                }

                if (obs_info.residual_id)
                {
                    observations.push_back(obs_info);
                    feature_indices.push_back(i);
                    num_valid_observations++;
                }
            }
        } // Release mutex here

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
            
            // Store initial pose parameters for resetting each round
            Eigen::Vector6d initial_pose_params = pose_params;
            
            for (int round = 0; round < config.m_outlier_detection_rounds; ++round)
            {
                // Reset pose to initial value for each round
                if (round > 0) {
                    pose_params = initial_pose_params;
                    // spdlog::debug("[POSE_OPT] Round {}: Reset pose to initial value", round);
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
                result.initial_cost = initial_cost;
                result.final_cost = summary.final_cost;
                result.num_iterations += summary.iterations.size();
                result.success = (summary.termination_type == ceres::CONVERGENCE);
            }
            
            // Print consolidated optimization summary
            double *pose_data = pose_params.data();
            int final_inliers = detect_outliers(const_cast<double const *const *>(&pose_data), observations, feature_indices, frame);
            int final_outliers = observations.size() - final_inliers;
            
            // spdlog::info("[POSE_OPT] {} rounds: cost {:.3e} -> {:.3e}, {} iters, {} inliers/{} outliers", 
            //             config.m_outlier_detection_rounds, initial_cost, final_cost, 
            //             total_iterations, final_inliers, final_outliers);
                        
            // Detailed Ceres summary logging removed
        }
        else
        {
            // Single solve without outlier detection rounds
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            if (config.m_enable_outlier_detection) {
                // Perform outlier detection for final report
                double *pose_data = pose_params.data();
                int num_inliers = detect_outliers(const_cast<double const *const *>(&pose_data), observations, feature_indices, frame);
                int num_outliers = observations.size() - num_inliers;
                
                // spdlog::info("[POSE_OPT] Single solve: cost {:.3e} -> {:.3e}, {} iters, {} inliers/{} outliers", 
                //             summary.initial_cost, summary.final_cost, summary.iterations.size(),
                //             num_inliers, num_outliers);
            } else {
                // No outlier detection
                // spdlog::info("[POSE_OPT] Single solve: cost {:.3e} -> {:.3e}, {} iters, ALL {} features treated as inliers", 
                //             summary.initial_cost, summary.final_cost, summary.iterations.size(),
                //             observations.size());
            }

            result.success = (summary.termination_type == ceres::CONVERGENCE);
            result.initial_cost = summary.initial_cost;
            result.final_cost = summary.final_cost;
            result.num_iterations = summary.iterations.size();
            
            // Detailed Ceres summary logging removed
        }

        // Count final inliers/outliers and disconnect outlier map points based on config
        if (config.m_enable_outlier_detection) {
            result.num_inliers = 0;
            result.num_outliers = 0;
            int disconnected_map_points = 0;
            
            // Protect MapPoint disconnection with mutex
            std::lock_guard<std::mutex> lock(s_mappoint_mutex);
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
            
            // if (disconnected_map_points > 0) {
            //     spdlog::warn("POSE_OPT[] Disconnected {} outlier map points", disconnected_map_points);
            // }
        } else {
            // Treat all observations as inliers when outlier detection is disabled
            result.num_inliers = observations.size();
            result.num_outliers = 0;
        }

        // Update result
        result.optimized_pose = se3_tangent_to_matrix(pose_params);


        // Update frame pose if optimization was successful
        if (result.success)
        {
            std::lock_guard<std::mutex> lock(s_keyframe_mutex);
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
    ObservationInfo PnPOptimizer::add_mono_observation(
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
        
        // Create mono PnP cost function with information matrix and T_cb
        auto cost_function = new factor::PnPFactor(observation, world_point, camera_params, T_cb, information);

        // Create robust loss function if enabled
        const Config& config = Config::getInstance();
        ceres::LossFunction *loss_function = nullptr;
        if (config.m_use_robust_kernel)
        {
            loss_function = create_robust_loss(sqrt(5.991));  // Chi-squared 95% threshold for 2 DOF
        }

        // Add residual block
        auto residual_id = problem.AddResidualBlock(
            cost_function, loss_function, pose_params);

        return ObservationInfo(residual_id, cost_function);
    }

    // Overloaded version with observation-based weighting
    ObservationInfo PnPOptimizer::add_mono_observation(
        ceres::Problem &problem,
        double *pose_params,
        const Eigen::Vector3d &world_point,
        const Eigen::Vector2d &observation,
        const factor::CameraParameters &camera_params,
        std::shared_ptr<Frame> frame,
        double pixel_noise_std,
        int num_observations)
    {

        // Create information matrix with observation-based weighting
        Eigen::Matrix2d information = create_information_matrix(pixel_noise_std, num_observations);

        // Get T_cb (body-to-camera transform) from frame directly - NO CONFIG ACCESS!
        const Eigen::Matrix4d& T_cb = frame->get_T_CB();
        
        // Create mono PnP cost function with information matrix and T_cb
        auto cost_function = new factor::PnPFactor(observation, world_point, camera_params, T_cb, information);

        // Create robust loss function if enabled
        const Config& config = Config::getInstance();
        ceres::LossFunction *loss_function = nullptr;
        if (config.m_use_robust_kernel)
        {
            loss_function = create_robust_loss(5.991);  // Chi-squared 95% threshold for 2 DOF
        }

        // Add residual block
        auto residual_id = problem.AddResidualBlock(
            cost_function, loss_function, pose_params);

        return ObservationInfo(residual_id, cost_function);
    }

    int PnPOptimizer::detect_outliers(double const *const *pose_params,
                                       const std::vector<ObservationInfo> &observations,
                                       const std::vector<int> &feature_indices,
                                       std::shared_ptr<Frame> frame)
    {
        int num_inliers = 0;

        // Chi-square threshold for 2DOF - use more relaxed threshold
        const double chi2_threshold = 5.991;  // Chi-square threshold for 2 DoF at 99% confidence

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
            
            // Get previous outlier status (protect with keyframe mutex)
            bool was_outlier;
            {
                std::lock_guard<std::mutex> lock(s_keyframe_mutex);
                was_outlier = frame->get_outlier_flag(feature_idx);
                
                // Update outlier flag - can be both set and cleared based on current chi2 test
                frame->set_outlier_flag(feature_idx, is_outlier);
            }

            // Set outlier flag in the cost function to disable it for next optimization round
            observations[i].cost_function->set_outlier(is_outlier);

            if (!is_outlier)
            {
                num_inliers++;
                inlier_chi2_values.push_back(chi2_error);
                
                // Log recovery if this feature was previously an outlier
                if (was_outlier) {
                    // spdlog::debug("[POSE_OPT] Feature {} recovered from outlier (chi2: {:.3f})", 
                    //              feature_idx, chi2_error);
                }
            }
            else
            {
                outlier_chi2_values.push_back(chi2_error);
                
                // Log new outlier detection
                if (!was_outlier) {
                    // spdlog::debug("[POSE_OPT] Feature {} marked as outlier (chi2: {:.3f})", feature_idx, chi2_error);
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

        // // Debug: Print details for some outliers to understand what's wrong
        // int debug_count = 0;
        // Eigen::Map<const Eigen::Vector6d> se3_tangent(pose_params[0]);
        // Sophus::SE3d current_pose = Sophus::SE3d::exp(se3_tangent);
        
        // // Convert matrix to string for logging
        // std::stringstream ss;
        // ss << current_pose.matrix();
        // // spdlog::debug("[OUTLIER_DEBUG] Current pose Twb:\n{}", ss.str());
        
        // for (size_t i = 0; i < observations.size() && debug_count < 3; ++i)
        // {
        //     double chi2_error = observations[i].cost_function->compute_chi_square(pose_params);
        //     if (chi2_error > chi2_threshold)
        //     {
        //         int feature_idx = feature_indices[i];
        //         auto feature = frame->get_features()[feature_idx];
        //         auto mp = frame->get_map_points()[feature_idx];
                
        //         // Manually project to see what the expected pixel should be
        //         Eigen::Vector3d world_pos = mp->get_position().cast<double>();
                
        //         // Transform to camera coordinates: Pc = Rcw * Pw + tcw
        //         Eigen::Matrix3d Rwb = current_pose.rotationMatrix();
        //         Eigen::Vector3d t_wb = current_pose.translation();
        //         Eigen::Matrix3d Rbw = Rwb.transpose();
        //         Eigen::Vector3d t_bw = -Rbw * t_wb;
                
        //         // Get T_cb (body-to-camera transform) from frame directly - CONSISTENT WITH add_mono_observation
        //         const Eigen::Matrix4d& T_cb = frame->get_T_CB();
                
        //         // Transform to camera coordinates: Pc = T_cb * (Rbw * Pw + t_bw)
        //         Eigen::Vector3d point_body = Rbw * world_pos + t_bw;
        //         Eigen::Vector4d point_body_h(point_body.x(), point_body.y(), point_body.z(), 1.0);
        //         Eigen::Vector4d point_camera_h = T_cb * point_body_h;
        //         Eigen::Vector3d point_camera = point_camera_h.head<3>();
                
        //         // Project to image plane
        //         double fx, fy, cx, cy;
        //         frame->get_camera_intrinsics(fx, fy, cx, cy);
                
        //         if (point_camera.z() > 0) {
        //             double u_proj = fx * point_camera.x() / point_camera.z() + cx;
        //             double v_proj = fy * point_camera.y() / point_camera.z() + cy;
                    
        //             // Get undistorted observation using normalized coordinates (consistent with CREATE_MP)
        //             cv::Point2f undistorted_pixel = feature->get_undistorted_coord();
        //             double undist_u = undistorted_pixel.x;
        //             double undist_v = undistorted_pixel.y;

        //             auto map_point = frame->get_map_point(feature_idx);
        //             // spdlog::debug("[OUTLIER_DEBUG] Feature {}: chi2={:.3f}, observed_undist=({:.1f},{:.1f}), projected=({:.1f},{:.1f}), world=({:.2f},{:.2f},{:.2f})", 
        //             //              feature_idx, chi2_error, undist_u, undist_v,
        //             //              u_proj, v_proj, world_pos.x(), world_pos.y(), world_pos.z());
        //         } else {
        //             // spdlog::debug("[OUTLIER_DEBUG] Feature {}: chi2={:.3f}, BEHIND_CAMERA: z={:.2f}", 
        //             //              feature_idx, chi2_error, point_camera.z());
        //         }
        //         debug_count++;
        //     }
        // }

        return num_inliers;
    }

    ceres::Solver::Options PnPOptimizer::setup_solver_options() const
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

        // Logging configuration - simplified (no config variables)
        options.logging_type = ceres::SILENT;
        options.minimizer_progress_to_stdout = false;

        return options;
    }

    Eigen::Vector6d PnPOptimizer::frame_to_se3_tangent(std::shared_ptr<Frame> frame) const
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

    Eigen::Matrix4f PnPOptimizer::se3_tangent_to_matrix(const Eigen::Vector6d &se3_tangent) const
    {
        // Convert tangent space to SE3 using Sophus (already guarantees proper SE3)
        Sophus::SE3d se3 = Sophus::SE3d::exp(se3_tangent);

        // Sophus already ensures proper SE3 structure, no need for SVD
        // Just convert to float at the end
        return se3.matrix().cast<float>();
    }

    ceres::LossFunction *PnPOptimizer::create_robust_loss(double delta) const
    {
        return new ceres::HuberLoss(delta);
    }

    Eigen::Matrix2d PnPOptimizer::create_information_matrix(double pixel_noise) const
    {
        // Information matrix is inverse of covariance matrix
        // For isotropic pixel noise: Covariance = sigma^2 * I
        // Information = (1/sigma^2) * I
        double precision = 1.0 / (pixel_noise * pixel_noise);
        return precision * Eigen::Matrix2d::Identity();
    }

    Eigen::Matrix2d PnPOptimizer::create_information_matrix(double pixel_noise, int num_observations) const
    {
        // Base precision from pixel noise
        double base_precision = 1.0 / (pixel_noise * pixel_noise);
        
        // Get config instance
        const Config& config = Config::getInstance();
        
        // Weight by number of observations (configurable max weight for PnP)
        double observation_weight = std::min(static_cast<double>(num_observations), config.m_pnp_max_observation_weight);
        
        // Final precision = base_precision * observation_weight
        double final_precision = base_precision * observation_weight;
        
        return final_precision * Eigen::Matrix2d::Identity();
    }

}

// SlidingWindowOptimizer implementation
namespace lightweight_vio {

// Define global mutexes for SlidingWindowOptimizer
std::mutex SlidingWindowOptimizer::s_mappoint_mutex;
std::mutex SlidingWindowOptimizer::s_keyframe_mutex;

SlidingWindowOptimizer::SlidingWindowOptimizer(size_t window_size)
    : m_window_size(window_size)
{
    // Get config for initialization
    const Config& config = Config::getInstance();
    m_max_iterations = config.m_sw_max_iterations;  // Use sliding window specific max iterations
    m_huber_delta = sqrt(5.991);                          // Chi-squared 95% threshold for 2 DOF (hardcoded)
    m_pixel_noise_std = 1.0;                        // Default pixel noise
    m_outlier_threshold = 5.991;                    // Chi-square threshold for 2 DoF at 98% confidence (more relaxed)
}

SlidingWindowResult SlidingWindowOptimizer::optimize(
    const std::vector<std::shared_ptr<Frame>>& keyframes,
    bool* force_stop_flag) {
    
    SlidingWindowResult result;
    
    if (keyframes.empty()) {
        spdlog::warn("[SlidingWindowOptimizer] No keyframes provided");
        return result;
    }
    
    if (keyframes.size() < 2) {
        spdlog::warn("[SlidingWindowOptimizer] Need at least 2 keyframes for bundle adjustment");
        return result;
    }
    
    // Collect map points observed by keyframes in sliding window
    auto map_points = collect_window_map_points(keyframes);
    
    if (map_points.empty()) {
        spdlog::warn("[SlidingWindowOptimizer] No map points found in sliding window");
        return result;
    }
    
    // Setup Ceres problem
    ceres::Problem problem;
    
    // Parameter storage
    std::vector<std::vector<double>> pose_params_vec(keyframes.size(), std::vector<double>(6));
    std::vector<std::vector<double>> point_params_vec(map_points.size(), std::vector<double>(3));
    
    // Setup optimization problem
    auto observations = setup_optimization_problem(
        problem, keyframes, map_points, pose_params_vec, point_params_vec);
    
    if (observations.empty()) {
        spdlog::warn("[SlidingWindowOptimizer] No valid observations found");
        return result;
    }
    
    // Apply marginalization strategy (fix oldest keyframe, etc.)
    apply_marginalization_strategy(
        problem, keyframes, map_points, pose_params_vec, point_params_vec);
    
    // Configure solver options
    ceres::Solver::Options first_stage_options = setup_solver_options();
    ceres::Solver::Options second_stage_options = setup_solver_options();
    
    // First stage: Use half iterations for quick outlier detection
    first_stage_options.max_num_iterations = std::max(1, m_max_iterations / 2);
    
    // Second stage: Use full iterations for precise optimization
    second_stage_options.max_num_iterations = m_max_iterations;
    
    if (force_stop_flag) {
        // TODO: Add custom callback for early termination if needed
    }
    
    ceres::Solver::Summary summary;
    
    // Initial cost evaluation
    double initial_cost = 0.0;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), &initial_cost, nullptr, nullptr, nullptr);
    result.initial_cost = initial_cost;
    
    // Stage 1: Quick optimization with robust kernel (half iterations)
    ceres::Solve(first_stage_options, &problem, &summary);
    
    if (force_stop_flag && *force_stop_flag) {
        result.success = false;
        return result;
    }
    
    // Outlier detection phase: mark outliers but don't remove them yet
    for (const auto& obs_info : observations) {
        const double* pose_params = pose_params_vec[obs_info.keyframe_index].data();
        const double* point_params = point_params_vec[obs_info.mappoint_index].data();
        const double* params[2] = {pose_params, point_params};
        
        // Compute chi-square error
        double chi_square = obs_info.cost_function->compute_chi_square(params);
        
        // Mark as outlier if above threshold (equivalent to g2o's setLevel(1))
        bool is_outlier = (chi_square > m_outlier_threshold);
        obs_info.cost_function->set_outlier(is_outlier);
    }
    
    // Stage 2: Precise optimization without robust kernel (full iterations)
    // Outliers are disabled via set_outlier - they return zero residuals
    ceres::Solver::Summary final_summary;
    ceres::Solve(second_stage_options, &problem, &final_summary);
    summary = final_summary; // Use final summary for cost reporting
    
    // Final cost evaluation
    double final_cost = 0.0;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), &final_cost, nullptr, nullptr, nullptr);
    result.final_cost = final_cost;
    result.num_iterations = summary.iterations.size();
    
    // Detect outliers and count inliers
    int num_inliers = detect_ba_outliers(
        pose_params_vec, point_params_vec, observations, keyframes, map_points);
    
    result.num_inliers = num_inliers;
    result.num_outliers = static_cast<int>(observations.size()) - num_inliers;
    result.num_poses_optimized = static_cast<int>(keyframes.size());
    result.num_points_optimized = static_cast<int>(map_points.size());
    
    // Check if optimization was successful - SUCCESS if cost decreased
    bool cost_decreased = (result.final_cost < result.initial_cost);
    result.success = cost_decreased;
    
    if (result.success) {
        // Update keyframes and map points with optimized values
        update_optimized_values(keyframes, map_points, pose_params_vec, point_params_vec);
        
        // spdlog::info("[SlidingWindowOptimizer] ✅ Optimization successful (cost decreased): {} poses, {} points, {} inliers, {} outliers, cost: {:.2e} -> {:.2e}",
        //             result.num_poses_optimized, result.num_points_optimized, 
        //             result.num_inliers, result.num_outliers, result.initial_cost, result.final_cost);
    } else {
        // spdlog::warn("[SlidingWindowOptimizer] ❌ Optimization failed (cost increased): {:.2e} -> {:.2e}, {}", 
        //             result.initial_cost, result.final_cost, summary.BriefReport());
    }
    
    return result;
}

std::vector<std::shared_ptr<MapPoint>> SlidingWindowOptimizer::collect_window_map_points(
    const std::vector<std::shared_ptr<Frame>>& keyframes) const {
    
    std::set<std::shared_ptr<MapPoint>> unique_map_points;
    
    // Collect all unique map points from keyframes with mutex protection
    {
        std::lock_guard<std::mutex> lock(s_mappoint_mutex);
        for (const auto& keyframe : keyframes) {
            if (!keyframe) continue;
            
            const auto& map_points = keyframe->get_map_points();
            for (const auto& mp : map_points) {
                if (mp && !mp->is_bad()) {
                    unique_map_points.insert(mp);
                }
            }
        }
    }
    
    // Convert set to vector
    std::vector<std::shared_ptr<MapPoint>> result(unique_map_points.begin(), unique_map_points.end());
    
    // spdlog::info("[SlidingWindowOptimizer] Collected {} unique map points from {} keyframes",
    //             result.size(), keyframes.size());
    
    return result;
}

BAObservationInfo SlidingWindowOptimizer::add_ba_observation(
    ceres::Problem& problem,
    double* pose_params,
    double* point_params,
    const Eigen::Vector2d& observation,
    const factor::CameraParameters& camera_params,
    std::shared_ptr<Frame> frame,
    int kf_index,
    int mp_index,
    double pixel_noise_std) {
    
    // Get T_CB transformation from frame
    Eigen::Matrix4d T_CB = frame->get_T_CB();
    
    // Create information matrix
    Eigen::Matrix2d information = create_information_matrix(pixel_noise_std);
    
    // Create BA factor
    auto* cost_function = new factor::BAFactor(
        observation, camera_params, T_CB, information);
    
    // Create robust loss function
    ceres::LossFunction* loss_function = create_robust_loss(m_huber_delta);
    
    // Add residual block to problem
    ceres::ResidualBlockId residual_id = problem.AddResidualBlock(
        cost_function, loss_function, pose_params, point_params);
    
    return BAObservationInfo(residual_id, cost_function, kf_index, mp_index);
}

BAObservationInfo SlidingWindowOptimizer::add_ba_observation(
    ceres::Problem& problem,
    double* pose_params,
    double* point_params,
    const Eigen::Vector2d& observation,
    const factor::CameraParameters& camera_params,
    std::shared_ptr<Frame> frame,
    int kf_index,
    int mp_index,
    double pixel_noise_std,
    int num_observations) {
    
    // Get T_CB transformation from frame
    Eigen::Matrix4d T_CB = frame->get_T_CB();
    
    // Create information matrix with observation-based weighting
    Eigen::Matrix2d information = create_information_matrix(pixel_noise_std, num_observations);
    
    // Create BA factor
    auto* cost_function = new factor::BAFactor(
        observation, camera_params, T_CB, information);
    
    // Create robust loss function
    ceres::LossFunction* loss_function = create_robust_loss(m_huber_delta);
    
    // Add residual block to problem
    ceres::ResidualBlockId residual_id = problem.AddResidualBlock(
        cost_function, loss_function, pose_params, point_params);
    
    return BAObservationInfo(residual_id, cost_function, kf_index, mp_index);
}

std::vector<BAObservationInfo> SlidingWindowOptimizer::setup_optimization_problem(
    ceres::Problem& problem,
    const std::vector<std::shared_ptr<Frame>>& keyframes,
    const std::vector<std::shared_ptr<MapPoint>>& map_points,
    std::vector<std::vector<double>>& pose_params_vec,
    std::vector<std::vector<double>>& point_params_vec) {
    
    std::vector<BAObservationInfo> observations;
    
    // Create map from MapPoint pointer to index for fast lookup
    std::unordered_map<std::shared_ptr<MapPoint>, int> mappoint_to_index;
    for (size_t i = 0; i < map_points.size(); ++i) {
        mappoint_to_index[map_points[i]] = static_cast<int>(i);
    }
    
    // Initialize pose parameters from keyframes with keyframe mutex protection
    {
        std::lock_guard<std::mutex> lock(s_keyframe_mutex);
        for (size_t kf_idx = 0; kf_idx < keyframes.size(); ++kf_idx) {
            const auto& keyframe = keyframes[kf_idx];
            Eigen::Matrix4f T_wb = keyframe->get_Twb();
            
            // Convert to double precision
            Eigen::Matrix4d T_wb_d = T_wb.cast<double>();
            
            // Extract rotation and translation
            Eigen::Matrix3d R_wb = T_wb_d.block<3, 3>(0, 0);
            Eigen::Vector3d t_wb = T_wb_d.block<3, 1>(0, 3);
            
            // Ensure rotation matrix is perfectly orthogonal using SVD
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(R_wb, Eigen::ComputeFullU | Eigen::ComputeFullV);
            R_wb = svd.matrixU() * svd.matrixV().transpose();
            
            // Ensure proper rotation (det = 1, not -1)
            if (R_wb.determinant() < 0) {
                Eigen::Matrix3d V_corrected = svd.matrixV();
                V_corrected.col(2) *= -1;  // Flip last column
                R_wb = svd.matrixU() * V_corrected.transpose();
            }
            
            // Reconstruct clean transformation matrix
            Eigen::Matrix4d T_wb_clean = Eigen::Matrix4d::Identity();
            T_wb_clean.block<3, 3>(0, 0) = R_wb;
            T_wb_clean.block<3, 1>(0, 3) = t_wb;
            
            // Convert to SE3 tangent space
            Sophus::SE3d se3_pose(T_wb_clean);
            Eigen::Vector6d tangent = se3_pose.log();
            
            std::copy(tangent.data(), tangent.data() + 6, pose_params_vec[kf_idx].data());
            
            // Add parameter block to problem FIRST
            problem.AddParameterBlock(pose_params_vec[kf_idx].data(), 6);
            
            // Then set pose parameterization
            auto* pose_parameterization = new factor::SE3GlobalParameterization();
            problem.SetParameterization(pose_params_vec[kf_idx].data(), pose_parameterization);
        }
    }
    
    // Initialize map point parameters
    for (size_t mp_idx = 0; mp_idx < map_points.size(); ++mp_idx) {
        const auto& map_point = map_points[mp_idx];
        Eigen::Vector3f position = map_point->get_position();
        
        point_params_vec[mp_idx][0] = position.x();
        point_params_vec[mp_idx][1] = position.y();
        point_params_vec[mp_idx][2] = position.z();
        
        // Add parameter block to problem FIRST
        problem.AddParameterBlock(point_params_vec[mp_idx].data(), 3);
        
        // Then set point parameterization
        auto* point_parameterization = new factor::MapPointParameterization();
        problem.SetParameterization(point_params_vec[mp_idx].data(), point_parameterization);
    }
    
    // Get camera parameters
    const Config& config = Config::getInstance();
    cv::Mat K = config.left_camera_matrix();
    factor::CameraParameters camera_params(
        K.at<double>(0, 0),  // fx
        K.at<double>(1, 1),  // fy
        K.at<double>(0, 2),  // cx
        K.at<double>(1, 2)   // cy
    );
    
    // Add observations for each keyframe with mutex protection
    {
        std::lock_guard<std::mutex> lock(s_mappoint_mutex);
        for (size_t kf_idx = 0; kf_idx < keyframes.size(); ++kf_idx) {
            const auto& keyframe = keyframes[kf_idx];
            const auto& features = keyframe->get_features();
            const auto& frame_map_points = keyframe->get_map_points();
            
            for (size_t feat_idx = 0; feat_idx < features.size(); ++feat_idx) {
                const auto& feature = features[feat_idx];
                const auto& map_point = frame_map_points[feat_idx];
                
                // Skip invalid features or map points
                if (!feature || !feature->is_valid() || !map_point || map_point->is_bad()) {
                    continue;
                }
                
                // Skip outlier features
                if (keyframe->get_outlier_flag(feat_idx)) {
                    continue;
                }
                
                // Find map point index
                auto it = mappoint_to_index.find(map_point);
                if (it == mappoint_to_index.end()) {
                    continue; // Map point not in our optimization set
                }
                
                int mp_idx = it->second;
                
                // Get 2D observation using undistorted coordinates (consistent with PnP optimizer)
                cv::Point2f undistorted_pixel = feature->get_undistorted_coord();
                Eigen::Vector2d observation(undistorted_pixel.x, undistorted_pixel.y);
                
                // Add BA observation with observation-based information weighting
                int num_observations = map_point->get_observation_count();
                auto obs_info = add_ba_observation(
                    problem,
                    pose_params_vec[kf_idx].data(),
                    point_params_vec[mp_idx].data(),
                    observation,
                    camera_params,
                    keyframe,
                    static_cast<int>(kf_idx),
                    mp_idx,
                    m_pixel_noise_std,
                    num_observations);
                
                observations.push_back(obs_info);
            }
        }
    }
    
    // spdlog::info("[SlidingWindowOptimizer] Setup problem: {} keyframes, {} map points, {} observations",
    //             keyframes.size(), map_points.size(), observations.size());
    
    return observations;
}

void SlidingWindowOptimizer::apply_marginalization_strategy(
    ceres::Problem& problem,
    const std::vector<std::shared_ptr<Frame>>& keyframes,
    const std::vector<std::shared_ptr<MapPoint>>& map_points,
    const std::vector<std::vector<double>>& pose_params_vec,
    const std::vector<std::vector<double>>& point_params_vec) {
    
    if (keyframes.empty()) return;
    
    // Fix the oldest keyframe (first in vector) as reference to prevent gauge freedom
    if (!pose_params_vec.empty()) {
        problem.SetParameterBlockConstant(const_cast<double*>(pose_params_vec[0].data()));
        // spdlog::debug("[SlidingWindowOptimizer] Fixed oldest keyframe {} as reference",
        //              keyframes[0]->get_frame_id());
    }
    
    // Optional: Fix map points with insufficient observations
    int fixed_points = 0;
    for (size_t mp_idx = 0; mp_idx < map_points.size(); ++mp_idx) {
        const auto& map_point = map_points[mp_idx];
        int obs_count = map_point->get_observation_count();
        
        // Fix map points with too few observations
        if (obs_count < 2) {
            problem.SetParameterBlockConstant(const_cast<double*>(point_params_vec[mp_idx].data()));
            fixed_points++;
        }
    }
    
    // if (fixed_points > 0) {
    //     spdlog::debug("[SlidingWindowOptimizer] Fixed {} / {} map points with insufficient observations",
    //                  fixed_points, map_points.size());
    // }
}

int SlidingWindowOptimizer::detect_ba_outliers(
    const std::vector<std::vector<double>>& pose_params_vec,
    const std::vector<std::vector<double>>& point_params_vec,
    const std::vector<BAObservationInfo>& observations,
    const std::vector<std::shared_ptr<Frame>>& keyframes,
    const std::vector<std::shared_ptr<MapPoint>>& map_points) {
    
    int num_inliers = 0;
    std::set<int> outlier_map_point_indices; // Track which map points are outliers
    std::vector<double> chi2_values;
    
    for (const auto& obs_info : observations) {
        // Get parameter pointers
        const double* pose_params = pose_params_vec[obs_info.keyframe_index].data();
        const double* point_params = point_params_vec[obs_info.mappoint_index].data();
        
        const double* params[2] = {pose_params, point_params};
        
        // Compute chi-square error
        double chi_square = obs_info.cost_function->compute_chi_square(params);
        chi2_values.push_back(chi_square);
        
        // Check against threshold
        bool is_inlier = (chi_square <= m_outlier_threshold);
        if (is_inlier) {
            num_inliers++;
        } else {
            // Mark this map point index as outlier
            outlier_map_point_indices.insert(obs_info.mappoint_index);
        }
        
        // Mark outlier in cost function (will return zero residuals)
        obs_info.cost_function->set_outlier(!is_inlier);
    }
    
    // Mark all outlier map points as bad and disconnect from all frames
    int marked_bad = 0;
    int disconnected_features = 0;
    
    // Protect MapPoint modifications and keyframe outlier flags with mutexes
    {
        std::lock_guard<std::mutex> mp_lock(s_mappoint_mutex);
        std::lock_guard<std::mutex> kf_lock(s_keyframe_mutex);
        
        for (int mp_idx : outlier_map_point_indices) {
            if (mp_idx >= 0 && mp_idx < static_cast<int>(map_points.size())) {
                auto map_point = map_points[mp_idx];
                if (map_point && !map_point->is_bad()) {
                    // First, find all frames that observe this map point and mark their features as outliers
                    for (const auto& keyframe : keyframes) {
                        const auto& frame_map_points = keyframe->get_map_points();
                        for (size_t feat_idx = 0; feat_idx < frame_map_points.size(); ++feat_idx) {
                            if (frame_map_points[feat_idx] == map_point) {
                                // Mark this feature as outlier in the frame
                                keyframe->set_outlier_flag(feat_idx, true);
                                // Remove the map point connection
                                keyframe->set_map_point(feat_idx, nullptr);
                                disconnected_features++;
                            }
                        }
                    }
                    
                    // Then mark the map point as bad
                    map_point->set_bad();
                    marked_bad++;
                }
            }
        }
    }
    
    // Log chi-square statistics
    if (!chi2_values.empty()) {
        auto minmax = std::minmax_element(chi2_values.begin(), chi2_values.end());
        double mean = std::accumulate(chi2_values.begin(), chi2_values.end(), 0.0) / chi2_values.size();
        
    }
    
   
    return num_inliers;
}

void SlidingWindowOptimizer::update_optimized_values(
    const std::vector<std::shared_ptr<Frame>>& keyframes,
    const std::vector<std::shared_ptr<MapPoint>>& map_points,
    const std::vector<std::vector<double>>& pose_params_vec,
    const std::vector<std::vector<double>>& point_params_vec) {
    
    int updated_keyframes = 0;
    int updated_map_points = 0;
    
    // Update keyframe poses with keyframe mutex protection
    {
        std::lock_guard<std::mutex> lock(s_keyframe_mutex);
        for (size_t kf_idx = 0; kf_idx < keyframes.size(); ++kf_idx) {
            const auto& keyframe = keyframes[kf_idx];
            if (!keyframe) continue;
            
            const auto& pose_params = pose_params_vec[kf_idx];
            
            // Store original pose for comparison
            Eigen::Matrix4f original_pose = keyframe->get_Twb();
            
            // Convert SE3 tangent space back to matrix
            Eigen::Map<const Eigen::Vector6d> tangent(pose_params.data());
            Sophus::SE3d se3_pose = Sophus::SE3d::exp(tangent);
            Eigen::Matrix4f T_wb = se3_pose.matrix().cast<float>();
            
            // Check if pose actually changed
            Eigen::Matrix4f pose_diff = T_wb - original_pose;
            double pose_change = pose_diff.norm();
            
            keyframe->set_Twb(T_wb);
            updated_keyframes++;
            
            // Log significant pose changes
            if (pose_change > 0.01) {
                spdlog::debug("[UPDATE] Keyframe {} pose changed by {:.4f}", 
                             keyframe->get_frame_id(), pose_change);
            }
        }
    }
    
    // Update map point positions with mutex protection
    {
        std::lock_guard<std::mutex> lock(s_mappoint_mutex);
        for (size_t mp_idx = 0; mp_idx < map_points.size(); ++mp_idx) {
            const auto& map_point = map_points[mp_idx];
            if (!map_point || map_point->is_bad()) continue;
            
            const auto& point_params = point_params_vec[mp_idx];
            
            // Store original position for comparison
            Eigen::Vector3f original_pos = map_point->get_position();
            
            Eigen::Vector3f new_position(
                static_cast<float>(point_params[0]),
                static_cast<float>(point_params[1]),
                static_cast<float>(point_params[2]));
            
            // Check if position actually changed
            Eigen::Vector3f pos_diff = new_position - original_pos;
            double position_change = pos_diff.norm();
            
            map_point->set_position(new_position);
            updated_map_points++;
            
            // // Log significant position changes
            // if (position_change > 0.1) {
            //     spdlog::debug("[UPDATE] MapPoint {} position changed by {:.4f}m", 
            //                  map_point->get_id(), position_change);
            // }
        }
    }
    
    // spdlog::info("[UPDATE] Updated {} keyframes and {} map points", 
    //             updated_keyframes, updated_map_points);
}

ceres::Solver::Options SlidingWindowOptimizer::setup_solver_options() const {
    ceres::Solver::Options options;
    const Config& config = Config::getInstance();
    
    // Use sparse solver for bundle adjustment
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.preconditioner_type = ceres::SCHUR_JACOBI;
    
    // Solver parameters from config
    options.max_num_iterations = config.m_sw_max_iterations;
    options.function_tolerance = config.m_sw_function_tolerance;
    options.gradient_tolerance = config.m_sw_gradient_tolerance;
    options.parameter_tolerance = config.m_sw_parameter_tolerance;
    
    // Enable detailed logging if needed
    options.minimizer_progress_to_stdout = false;
    options.logging_type = ceres::SILENT;
    
    // Use multiple threads if available
    options.num_threads = std::min(4, static_cast<int>(std::thread::hardware_concurrency()));
    
    return options;
}

ceres::LossFunction* SlidingWindowOptimizer::create_robust_loss(double delta) const {
    return new ceres::HuberLoss(delta);
}

Eigen::Matrix2d SlidingWindowOptimizer::create_information_matrix(double pixel_noise) const {
    Eigen::Matrix2d information_matrix;
    double variance = pixel_noise * pixel_noise;
    information_matrix << 1.0 / variance, 0.0,
                         0.0, 1.0 / variance;
    return information_matrix;
}

Eigen::Matrix2d SlidingWindowOptimizer::create_information_matrix(double pixel_noise, int num_observations) const {
    Eigen::Matrix2d information_matrix;
    double variance = pixel_noise * pixel_noise;
    
    // Get config instance
    const Config& config = Config::getInstance();
    
    // Weight by number of observations (configurable max weight for sliding window)
    double observation_weight = std::min(static_cast<double>(num_observations), config.m_sw_max_observation_weight);
    
    // Apply weighting to precision
    double weighted_precision = observation_weight / variance;
    
    information_matrix << weighted_precision, 0.0,
                         0.0, weighted_precision;
    return information_matrix;
}

} // namespace lightweight_vio
