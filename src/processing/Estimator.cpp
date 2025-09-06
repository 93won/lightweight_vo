#include <processing/Estimator.h>
#include <processing/FeatureTracker.h>
#include <database/Frame.h>
#include <database/MapPoint.h>
#include <processing/Optimizer.h>

#include <util/Config.h>
#include <util/EurocUtils.h>
#include <spdlog/spdlog.h>
#include <chrono>
#include <iostream>
#include <numeric>
#include <algorithm>

namespace lightweight_vio {

Estimator::Estimator()
    : m_frame_id_counter(0)
    , m_frames_since_last_keyframe(0)
    , m_last_keyframe_grid_coverage(0.0)
    , m_current_pose(Eigen::Matrix4f::Identity())
    , m_transform_from_last(Eigen::Matrix4f::Identity())
    , m_has_initial_gt_pose(false)
    , m_initial_gt_pose(Eigen::Matrix4f::Identity())
    , m_sliding_window_thread_running(false)
    , m_keyframes_updated(false) {
    
    // Initialize feature tracker
    m_feature_tracker = std::make_unique<FeatureTracker>();
    
    // Initialize pose optimizer - now uses global Config internally
    m_pose_optimizer = std::make_unique<PnPOptimizer>();
    
    // Initialize sliding window optimizer
    m_sliding_window_optimizer = std::make_unique<SlidingWindowOptimizer>(
        Config::getInstance().m_keyframe_window_size);  // window size only
    
    // Start sliding window optimization thread
    m_sliding_window_thread_running = true;
    m_sliding_window_thread = std::make_unique<std::thread>(&Estimator::sliding_window_thread_function, this);
    
    spdlog::info("[ESTIMATOR] Sliding window optimization thread started");
}

Estimator::EstimationResult Estimator::process_frame(const cv::Mat& left_image, const cv::Mat& right_image, long long timestamp) {
    EstimationResult result;
    auto total_start_time = std::chrono::high_resolution_clock::now();

    // Frame processing starts
    std::cout<<"\n";
    spdlog::info("============================== Frame {} ==============================\n", m_frame_id_counter);

    // Increment frame counter since last keyframe for every new frame
    m_frames_since_last_keyframe++;

    // Initialize timing variables
    double frame_creation_time = 0.0;
    double prediction_time = 0.0;
    double tracking_time = 0.0;
    double optimization_time = 0.0;

    // Create new stereo frame
    auto frame_creation_start = std::chrono::high_resolution_clock::now();
    m_current_frame = create_frame(left_image, right_image, timestamp);
    auto frame_creation_end = std::chrono::high_resolution_clock::now();
    frame_creation_time = std::chrono::duration_cast<std::chrono::microseconds>(frame_creation_end - frame_creation_start).count() / 1000.0;

    if (!m_current_frame)
    {
        spdlog::error("[Estimator] Failed to create frame!");
        result.success = false;
        return result;
    }

    if (m_previous_frame) {
        auto prediction_start = std::chrono::high_resolution_clock::now();
        predict_state();
        auto prediction_end = std::chrono::high_resolution_clock::now();
        prediction_time = std::chrono::duration_cast<std::chrono::microseconds>(prediction_end - prediction_start).count() / 1000.0;
        
        // Track features from previous frame using FeatureTracker
        // FeatureTracker now handles both tracking and map point association/creation
        auto tracking_start = std::chrono::high_resolution_clock::now();
        m_feature_tracker->track_features(m_current_frame, m_previous_frame);
        auto tracking_end = std::chrono::high_resolution_clock::now();
        tracking_time = std::chrono::duration_cast<std::chrono::microseconds>(tracking_end - tracking_start).count() / 1000.0;
        
        result.num_features = m_current_frame->get_feature_count();
        
        // Compute stereo depth for all features
        m_current_frame->compute_stereo_depth();
        
        // Count how many features have associated map points (already done by FeatureTracker)
        int num_tracked_with_map_points = count_features_with_map_points(m_current_frame);
        
        // Log tracking information
        spdlog::info("[TRACKING] {} features tracked, {} with map points", 
                    result.num_features, num_tracked_with_map_points);
        
        if (num_tracked_with_map_points > 0) {
            // ✅ ENABLED: Pose optimization re-enabled after fixing coordinate space mismatch!
            if (num_tracked_with_map_points >= 5) {
                auto optimization_start = std::chrono::high_resolution_clock::now();
                auto opt_result = optimize_pose(m_current_frame);
                auto optimization_end = std::chrono::high_resolution_clock::now();
                optimization_time = std::chrono::duration_cast<std::chrono::microseconds>(optimization_end - optimization_start).count() / 1000.0;
                
                result.success = opt_result.success;
                result.num_inliers = opt_result.num_inliers;
                result.num_outliers = opt_result.num_outliers;
                
                if (opt_result.success) {
                    m_current_pose = opt_result.optimized_pose;
                    m_current_frame->set_Twb(m_current_pose);
                    
                    // Update transform from last frame for velocity estimation
                    update_transform_from_last();
                    
                    spdlog::info("[POSE_OPT] ✅ Optimization successful: {} inliers, {} outliers, initial_cost={:.2f}, final_cost={:.2f}, iterations={}, time={:.2f}ms", 
                                opt_result.num_inliers, opt_result.num_outliers, opt_result.initial_cost, opt_result.final_cost, opt_result.num_iterations, optimization_time);
                } else {
                    spdlog::warn("[POSE_OPT] ❌ Optimization failed - keeping previous pose");
                }
            } else {
                spdlog::warn("[POSE_OPT] ⚠️ Not enough map point associations for optimization: {} (need ≥5)", num_tracked_with_map_points);
                // Fallback: use current pose as-is
                m_current_pose = m_current_frame->get_Twb();
                
                // Update transform from last frame for velocity estimation
                update_transform_from_last();
                
                result.success = true;
                result.num_inliers = num_tracked_with_map_points;
                result.num_outliers = 0;
            } 
        } else {
            // No tracking, keep previous pose (already set in create_frame)
            m_current_pose = m_current_frame->get_Twb();
            
            // Update transform from last frame for velocity estimation (even if tracking failed)
            update_transform_from_last();
            
            result.success = false;
        }

        // NOTE: FeatureTracker already handles map point association during tracking
        // No need to call associate_tracked_features_with_map_points() again
        
        
        // Decide whether to create keyframe
        auto keyframe_decision_start = std::chrono::high_resolution_clock::now();
        bool is_keyframe = should_create_keyframe(m_current_frame);
        auto keyframe_decision_end = std::chrono::high_resolution_clock::now();
        auto keyframe_decision_time = std::chrono::duration_cast<std::chrono::microseconds>(keyframe_decision_end - keyframe_decision_start).count() / 1000.0;
        
        // Only create new map points for keyframes to avoid trajectory drift
        if (is_keyframe) {
            auto map_points_start = std::chrono::high_resolution_clock::now();
            int new_map_points = create_new_map_points(m_current_frame);
            auto map_points_end = std::chrono::high_resolution_clock::now();
            auto map_points_time = std::chrono::duration_cast<std::chrono::microseconds>(map_points_end - map_points_start).count() / 1000.0;
            
            result.num_new_map_points = new_map_points;
            // spdlog::info("[MAP_POINTS] Created {} new map points by new keyframe insertion", new_map_points);
            
            auto keyframe_creation_start = std::chrono::high_resolution_clock::now();
            create_keyframe(m_current_frame);
            auto keyframe_creation_end = std::chrono::high_resolution_clock::now();
            auto keyframe_creation_time = std::chrono::duration_cast<std::chrono::microseconds>(keyframe_creation_end - keyframe_creation_start).count() / 1000.0;
            
            m_frames_since_last_keyframe = 0;  // Reset to 0 after creating keyframe
            
        } else {
            result.num_new_map_points = 0;
        }
        
        // Count tracked features and features with map points
        result.num_tracked_features = m_current_frame->get_feature_count();
        result.num_features_with_map_points = count_features_with_map_points(m_current_frame);
        
        // Compute reprojection error statistics for keyframes
        if (is_keyframe && count_features_with_map_points(m_current_frame) > 5) {
            compute_reprojection_error_statistics(m_current_frame);
        }
        
      
    } else {
        // First frame - extract features using FeatureTracker
        m_feature_tracker->track_features(m_current_frame, nullptr);
        
        result.num_features = m_current_frame->get_feature_count();
        
        // Compute stereo depth for all features
        auto stereo_start = std::chrono::high_resolution_clock::now();
        m_current_frame->compute_stereo_depth();
        auto stereo_end = std::chrono::high_resolution_clock::now();
        auto stereo_time = std::chrono::duration_cast<std::chrono::microseconds>(stereo_end - stereo_start).count() / 1000.0;
        
        // First frame - keep identity pose (already set in create_frame)
        m_current_pose = m_current_frame->get_Twb();
        
        // Increment frame counter (first frame processing)
        m_frames_since_last_keyframe++;
        
        // Create initial map points (first frame is always considered keyframe)
        auto initial_map_points_start = std::chrono::high_resolution_clock::now();
        int initial_map_points = create_initial_map_points(m_current_frame);
        auto initial_map_points_end = std::chrono::high_resolution_clock::now();
        auto initial_map_points_time = std::chrono::duration_cast<std::chrono::microseconds>(initial_map_points_end - initial_map_points_start).count() / 1000.0;
        
        result.num_new_map_points = initial_map_points;
        spdlog::info("[MAP_POINTS] Created {} initial map points", initial_map_points);
        
        auto first_keyframe_start = std::chrono::high_resolution_clock::now();
        create_keyframe(m_current_frame);
        auto first_keyframe_end = std::chrono::high_resolution_clock::now();
        auto first_keyframe_time = std::chrono::duration_cast<std::chrono::microseconds>(first_keyframe_end - first_keyframe_start).count() / 1000.0;
        
        m_frames_since_last_keyframe = 0;  // Reset after creating first keyframe
        
        spdlog::info("[TIMING] First frame initialization: stereo={:.2f}ms, initial_map_points={:.2f}ms, keyframe={:.2f}ms", 
                    stereo_time, initial_map_points_time, first_keyframe_time);
        
        // Count features for first frame
        result.num_tracked_features = m_current_frame->get_feature_count();
        result.num_features_with_map_points = count_features_with_map_points(m_current_frame);
        
        result.success = true;
    }
    
    // Update result
    result.pose = m_current_frame->get_Twb();
    
    // Add processed frame to all frames vector for trajectory export
    m_all_frames.push_back(m_current_frame);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - total_start_time);
    result.optimization_time_ms = duration.count() / 1000.0;
    
    
    // Update state
    m_previous_frame = m_current_frame;
    
    return result;
}

Estimator::~Estimator() {
    // Stop sliding window thread
    if (m_sliding_window_thread_running) {
        m_sliding_window_thread_running = false;
        m_keyframes_cv.notify_one();
        
        if (m_sliding_window_thread && m_sliding_window_thread->joinable()) {
            m_sliding_window_thread->join();
        }
        
        spdlog::info("[ESTIMATOR] Sliding window optimization thread stopped");
    }
}

void Estimator::reset() {
    m_current_frame.reset();
    m_previous_frame.reset();
    m_keyframes.clear();
    m_all_frames.clear();
    m_map_points.clear();
    
    m_frame_id_counter = 0;
    m_frames_since_last_keyframe = 0;
    m_last_keyframe_grid_coverage = 0.0;
    m_current_pose = Eigen::Matrix4f::Identity();
}

Eigen::Matrix4f Estimator::get_current_pose() const {
    return m_current_pose;
}

std::vector<std::shared_ptr<Frame>> Estimator::get_keyframes_safe() const {
    std::lock_guard<std::mutex> lock(m_keyframes_mutex);
    return m_keyframes;  // Return a copy
}

std::shared_ptr<Frame> Estimator::create_frame(const cv::Mat& left_image, const cv::Mat& right_image, long long timestamp) {
    if (left_image.empty() || right_image.empty()) {
        return nullptr;
    }
    
    // Convert to grayscale if needed
    cv::Mat gray_left, gray_right;
    if (left_image.channels() == 3) {
        cv::cvtColor(left_image, gray_left, cv::COLOR_BGR2GRAY);
    } else {
        gray_left = left_image.clone();
    }
    
    if (right_image.channels() == 3) {
        cv::cvtColor(right_image, gray_right, cv::COLOR_BGR2GRAY);
    } else {
        gray_right = right_image.clone();
    }
    
    // Create frame with stereo images and camera parameters
    // Get camera parameters from global config
    const auto& global_config = Config::getInstance();
    cv::Mat left_K = global_config.left_camera_matrix();
    
    auto frame = std::make_shared<Frame>(
        timestamp, 
        m_frame_id_counter++,
        gray_left, gray_right,
        left_K.at<double>(0, 0), left_K.at<double>(1, 1), left_K.at<double>(0, 2), left_K.at<double>(1, 2),
        global_config.left_dist_coeffs()
    );
    
    // Set initial pose
    if (m_previous_frame) {
        // For non-first frames, start with previous frame pose
        // Actual prediction will be done in process_frame() via predict_state()
        frame->set_Twb(m_previous_frame->get_Twb());
    } else {
        // First frame - use ground truth pose if available, otherwise identity
        if (m_has_initial_gt_pose) {
            frame->set_Twb(m_initial_gt_pose);
            spdlog::info("[GT_INIT] Initialized first frame with ground truth pose");
        } else {
            frame->set_Twb(Eigen::Matrix4f::Identity());
        }
    }
    
    return frame;
}



int lightweight_vio::Estimator::create_initial_map_points(std::shared_ptr<Frame> frame) {
    if (!frame) {
        return 0;
    }
    
    int num_created = 0;
    const auto& features = frame->get_features();
    
    // Create map points from stereo triangulated features
    for (size_t i = 0; i < features.size(); ++i) {
        auto feature = features[i];
        if (feature && feature->is_valid() && frame->has_depth(i)) {
            // Get 3D point in camera frame from stereo triangulation
            Eigen::Vector3f camera_3d_point = feature->get_3d_point();
            if (camera_3d_point.isZero()) {
                continue;  // Skip if no valid 3D point
            }
            
            // Transform to world coordinates using frame pose
            Eigen::Matrix4f T_wb = frame->get_Twb();
            
            // Use Frame's cached T_cb for consistency
            Eigen::Matrix4f T_cb = frame->get_T_CB().cast<float>();  // Body to Camera
            Eigen::Matrix4f T_bc = T_cb.inverse();     // Camera to Body
            
            // Transform: Camera → Body → World
            Eigen::Vector4f camera_point(camera_3d_point.x(), camera_3d_point.y(), camera_3d_point.z(), 1.0);
            Eigen::Vector4f body_point = T_bc * camera_point;  // Camera to Body
            Eigen::Vector4f world_point = T_wb * body_point;    // Body to World
            
            Eigen::Vector3f world_pos = world_point.head<3>();
            
            auto map_point = std::make_shared<MapPoint>(world_pos);
            m_map_points.push_back(map_point);
            
            // Associate with frame
            frame->set_map_point(i, map_point);
            num_created++;
            
            // Compute reprojection error for verification
            double fx, fy, cx, cy;
            frame->get_camera_intrinsics(fx, fy, cx, cy);
            
            // Project world point back to camera
            Eigen::Vector4f world_pos_h(world_pos.x(), world_pos.y(), world_pos.z(), 1.0f);
            Eigen::Vector4f camera_pos_h = T_cb * (T_wb.inverse() * world_pos_h);
            Eigen::Vector3f camera_pos = camera_pos_h.head<3>();
            
            if (camera_pos.z() > 0) {
                // Project to pixel coordinates using camera intrinsics
                float u_proj = fx * camera_pos.x() / camera_pos.z() + cx;
                float v_proj = fy * camera_pos.y() / camera_pos.z() + cy;
                
                // Get original undistorted pixel coordinates for fair comparison
                cv::Point2f undistorted_pixel = feature->get_undistorted_coord();
                // Convert normalized to undistorted pixel coordinates
                float undist_u = undistorted_pixel.x;
                float undist_v = undistorted_pixel.y;
                
                // Compute reprojection error in undistorted pixel coordinate space
                double error_x = undist_u - u_proj;
                double error_y = undist_v - v_proj;
                double error = std::sqrt(error_x * error_x + error_y * error_y);
            } else {
                // Point behind camera
            }
        }
    }
    
    spdlog::info("[MAP_POINTS] Created {} new map points from {} features", num_created, features.size());
    return num_created;
}



int lightweight_vio::Estimator::create_new_map_points(std::shared_ptr<Frame> frame) {
    if (!frame) {
        return 0;
    }
    
    int num_created = 0;
    const auto& features = frame->get_features();
    
    // Only create map points for features that:
    // 1. Have valid stereo depth
    // 2. Are not already associated with a map point
    // 3. Are valid features
    // 4. NEW: Include outlier features if they have stereo matches
    for (size_t i = 0; i < features.size(); ++i) {
        auto feature = features[i];
        
        if (!feature || !feature->is_valid()) {
            continue;
        }
        
        if (!frame->has_depth(i)) {
            continue;
        }
        
        if (frame->has_map_point(i)) {
            continue;
        }
        
        double depth = frame->get_depth(i);
        
        // Validate depth range using global config parameters
        auto& global_config = lightweight_vio::Config::getInstance();
        if (depth < global_config.m_min_depth || depth > global_config.m_max_depth) {
            continue;  // Skip invalid depths
        }
        
        // Get 3D point in camera frame from stereo triangulation
        Eigen::Vector3f camera_3d_point = feature->get_3d_point();
        if (camera_3d_point.isZero()) {
            continue;  // Skip if no valid 3D point
        }
        
        // Transform to world coordinates using frame pose
        Eigen::Matrix4f T_wb = frame->get_Twb();
        
        // Get actual T_bc from configuration 
        const auto& config = lightweight_vio::Config::getInstance();
        cv::Mat T_bc_cv = config.left_T_BC();  // Get T_BC from config (camera to body)
        
        Eigen::Matrix4f T_bc;
        if (!T_bc_cv.empty()) {
            // Convert T_bc to Eigen
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    T_bc(i, j) = T_bc_cv.at<double>(i, j);
                }
            }
        } else {
            // Fallback to identity if config not available
            T_bc = Eigen::Matrix4f::Identity();
        }
        
        // Transform: Camera → Body → World
        Eigen::Vector4f camera_point(camera_3d_point.x(), camera_3d_point.y(), camera_3d_point.z(), 1.0);
        Eigen::Vector4f body_point = T_bc * camera_point;
        Eigen::Vector4f world_point = T_wb * body_point;
        
        Eigen::Vector3f world_pos = world_point.head<3>();
        
        // Create new map point
        auto map_point = std::make_shared<MapPoint>(world_pos);
        m_map_points.push_back(map_point);
        
        // Associate with current frame
        frame->set_map_point(i, map_point);
        
        // Real-time reprojection verification for new map point
        // spdlog::debug("NEW Map Point {}: world_pos=({:.3f}, {:.3f}, {:.3f})", 
        //              m_map_points.size() - 1, world_pos.x(), world_pos.y(), world_pos.z());
        
        // Verify reprojection in current frame
        Eigen::Matrix4f T_wc = T_wb * T_bc;  // Camera to world
        Eigen::Matrix4f T_cw = T_wc.inverse(); // World to camera (for projection)
        
        // Project back to image
        Eigen::Vector4f world_homogeneous(world_pos.x(), world_pos.y(), world_pos.z(), 1.0);
        Eigen::Vector4f camera_projected = T_cw * world_homogeneous;
        
        if (camera_projected.z() > 0) {  // Valid projection
            auto feature = frame->get_feature(i);
            if (feature) {
                cv::Point2f observed_pt = feature->get_undistorted_coord();
                
                // Get camera intrinsics from frame
                double fx, fy, cx, cy;
                frame->get_camera_intrinsics(fx, fy, cx, cy);
                
                float projected_x = (fx * camera_projected.x() / camera_projected.z()) + cx;
                float projected_y = (fy * camera_projected.y() / camera_projected.z()) + cy;
                
                float reprojection_error = sqrt(pow(observed_pt.x - projected_x, 2) + pow(observed_pt.y - projected_y, 2));
                
                // spdlog::debug("  Reprojection: observed=({:.2f}, {:.2f}), projected=({:.2f}, {:.2f}), error={:.3f}px", 
                //              observed_pt.x, observed_pt.y, projected_x, projected_y, reprojection_error);
            }
        }
        
        num_created++;
    }
    
    return num_created;
}



bool lightweight_vio::Estimator::should_create_keyframe(std::shared_ptr<Frame> frame) {

    if (!frame) {
        return false;
    }
    
    // First frame is always a keyframe
    if (m_keyframes.empty()) {
        return true;
    }
    
    // Grid-based keyframe creation policy
    // Create keyframe when grid coverage drops to configured ratio of last keyframe's coverage
    double current_grid_coverage = calculate_grid_coverage_with_map_points(frame);
    
    // For the first few keyframes, use absolute threshold of 50% to establish baseline
    if (m_keyframes.size() <= 2 || m_last_keyframe_grid_coverage <= 0.0) {
        double absolute_threshold = 0.5;
        if (current_grid_coverage < absolute_threshold) {
            if (Config::getInstance().m_enable_debug_output) {
                spdlog::info("[KEYFRAME] Creating keyframe due to low grid coverage (initial): {:.2f} < {:.2f}", 
                            current_grid_coverage, absolute_threshold);
            }
            return true;
        }
    } else {
        // Use relative threshold based on last keyframe's coverage
        double relative_threshold = m_last_keyframe_grid_coverage * Config::getInstance().m_grid_coverage_ratio;
        if (current_grid_coverage < relative_threshold) {
            return true;
        }
    }
    
    return false;
}

void lightweight_vio::Estimator::create_keyframe(std::shared_ptr<Frame> frame) {
    if (!frame) {
        return;
    }
    
    frame->set_keyframe(true);
    
    // Thread-safe keyframe management
    {
        std::lock_guard<std::mutex> lock(m_keyframes_mutex);
        m_keyframes.push_back(frame);
        
        // Apply sliding window - remove old keyframes if window size exceeded
        const int max_keyframes = Config::getInstance().m_keyframe_window_size;
        if (m_keyframes.size() > static_cast<size_t>(max_keyframes)) {
            // Remove oldest keyframe
            auto oldest_keyframe = m_keyframes.front();
            
            // Clean up observations from map points before removing the keyframe
            int removed_observations = 0;
            const auto& features = oldest_keyframe->get_features();
            for (size_t i = 0; i < features.size(); ++i) {
                auto feature = features[i];
                auto map_point = oldest_keyframe->get_map_point(i);
                
                if (feature && feature->is_valid() && map_point && !map_point->is_bad()) {
                    // Remove observation from map point
                    map_point->remove_observation(oldest_keyframe);
                    removed_observations++;
                    
                    // Check if map point has no more observations after removal
                    if (map_point->get_observation_count() == 0) {
                        // Mark map point as bad if it has no observations
                        map_point->set_bad();
                    }
                }
            }
            
            m_keyframes.erase(m_keyframes.begin());
        }
    } // Release mutex lock here
    
    // 🎯 Update track count only when frame becomes keyframe
    const auto& features = frame->get_features();
    for (auto& feature : features) {
        if (feature && feature->is_valid()) {
            feature->increment_track_count();
        }
    }
    
    // Add observations to map points for this keyframe
    int observations_added = 0;
    // Use the existing features variable from above - no duplicate declaration
    for (size_t i = 0; i < features.size(); ++i) {
        auto feature = features[i];
        auto map_point = frame->get_map_point(i);
        
        if (feature && feature->is_valid() && map_point && !map_point->is_bad()) {
            // Add this keyframe as an observation to the map point
            map_point->add_observation(frame, i);
            observations_added++;
        }
    }
    
    // Store grid coverage of this keyframe for future relative comparisons
    m_last_keyframe_grid_coverage = calculate_grid_coverage_with_map_points(frame);
    
    // Notify sliding window optimization thread
    notify_sliding_window_thread();
}

OptimizationResult lightweight_vio::Estimator::optimize_pose(std::shared_ptr<Frame> frame) {
    // Create pose optimizer - uses global Config internally
    PnPOptimizer optimizer;
    return optimizer.optimize_pose(frame);
}

int lightweight_vio::Estimator::count_features_with_map_points(std::shared_ptr<Frame> frame) {
    if (!frame) {
        return 0;
    }
    
    int count = 0;
    const auto& map_points = frame->get_map_points();
    
    for (const auto& mp : map_points) {
        if (mp && !mp->is_bad()) {
            count++;
        }
    }
    
    return count;
}



void Estimator::set_initial_gt_pose(const Eigen::Matrix4f& gt_pose) {
    m_initial_gt_pose = gt_pose;
    m_has_initial_gt_pose = true;
    spdlog::info("[GT_INIT] Set initial ground truth pose");
}

void Estimator::apply_gt_pose_to_current_frame(const Eigen::Matrix4f& gt_pose) {
    if (m_current_frame) {
        m_current_frame->set_Twb(gt_pose);
        m_current_pose = gt_pose;
        spdlog::debug("[GT_APPLY] Applied GT pose to frame {}", m_current_frame->get_frame_id());
    }
}



void lightweight_vio::Estimator::compute_reprojection_error_statistics(std::shared_ptr<Frame> frame) {
    if (!frame) {
        return;
    }
    
    std::vector<double> reprojection_errors;
    const auto& features = frame->get_features();
    const auto& map_points = frame->get_map_points();
    
    // Get camera parameters
    double fx, fy, cx, cy;
    frame->get_camera_intrinsics(fx, fy, cx, cy);
    
    // DEBUG: Print camera parameters
    // spdlog::debug("[REPROJ_DEBUG] Camera params: fx={:.2f}, fy={:.2f}, cx={:.2f}, cy={:.2f}", fx, fy, cx, cy);
    
    // Get current pose
    Eigen::Matrix4f T_wb = frame->get_Twb();
    
    
    // Get T_cb from frame
    const Eigen::Matrix4d& T_cb = frame->get_T_CB();
    Eigen::Matrix4f T_cb_f = T_cb.cast<float>();
   
    
    int valid_projections = 0;
    int behind_camera = 0;
    
    for (size_t i = 0; i < features.size() && i < map_points.size(); ++i) {
        auto feature = features[i];
        auto map_point = map_points[i];
        
        if (!feature || !feature->is_valid() || !map_point || map_point->is_bad()) {
            continue;
        }
        
        // Get 3D world position
        Eigen::Vector3f world_pos = map_point->get_position();
        
        // CORRECTED: Transform from world to camera using correct matrix chain
        // T_wc = T_wb * T_bc (Camera → World)
        Eigen::Matrix4f T_bc = T_cb_f.inverse();  // Camera to Body (T_bc = T_cb^-1)
        Eigen::Matrix4f T_wc = T_wb * T_bc;   // Camera to World transformation
        
        // Transform world point to camera
        Eigen::Vector4f world_pos_h(world_pos.x(), world_pos.y(), world_pos.z(), 1.0f);
        Eigen::Vector4f camera_pos_h = T_wc.inverse() * world_pos_h;
        Eigen::Vector3f camera_pos = camera_pos_h.head<3>();
        
        // Check if point is behind camera
        if (camera_pos.z() <= 0) {
            behind_camera++;
            continue;
        }
        
        // Project to pixel coordinates using camera intrinsics
        float u_proj = fx * camera_pos.x() / camera_pos.z() + cx;
        float v_proj = fy * camera_pos.y() / camera_pos.z() + cy;
        
        // Get original undistorted pixel coordinates for fair comparison
        cv::Point2f undistorted_pixel = feature->get_undistorted_coord();
        // Convert normalized to undistorted pixel coordinates
        float undist_u = undistorted_pixel.x;
        float undist_v = undistorted_pixel.y;

        // Compute reprojection error in undistorted pixel coordinate space
        double error_x = undist_u - u_proj;
        double error_y = undist_v - v_proj;
        double error = std::sqrt(error_x * error_x + error_y * error_y);
        
        reprojection_errors.push_back(error);
        valid_projections++;
        
        // Print core info for all features in one line  
        // spdlog::info("[REPROJ] F{}: obs_undist=({:.1f},{:.1f}) proj_pixel=({:.1f},{:.1f}) err={:.2f}px world=({:.2f},{:.2f},{:.2f})", 
        //              i, undist_u, undist_v, u_proj, v_proj, error,
        //              world_pos.x(), world_pos.y(), world_pos.z());
    }
    
    if (valid_projections > 0) {
        // Compute statistics
        std::sort(reprojection_errors.begin(), reprojection_errors.end());
        
        double mean_error = std::accumulate(reprojection_errors.begin(), reprojection_errors.end(), 0.0) / reprojection_errors.size();
        double median_error = reprojection_errors[reprojection_errors.size() / 2];
        double min_error = reprojection_errors.front();
        double max_error = reprojection_errors.back();
        
        // Count outliers (error > 5.0 pixels in undistorted pixel space)
        int outliers = std::count_if(reprojection_errors.begin(), reprojection_errors.end(), 
                                   [](double error) { return error > 5.0; });
        
        // spdlog::info("[REPROJ_ERROR] Frame {}: {}/{} valid, mean={:.2f}px, median={:.2f}px, min={:.2f}px, max={:.2f}px, outliers={}", 
        //             frame->get_frame_id(), valid_projections, features.size(), 
        //             mean_error, median_error, min_error, max_error, outliers);
        
        if (behind_camera > 0) {
            spdlog::warn("[REPROJ_ERROR] {} points behind camera", behind_camera);
        }
    } else {
        spdlog::warn("[REPROJ_ERROR] Frame {}: No valid projections", frame->get_frame_id());
    }
}

void Estimator::predict_state() {
    if (m_current_frame && m_previous_frame) {
        Eigen::Matrix4f predicted_pose = m_previous_frame->get_Twb() * m_transform_from_last;
        m_current_frame->set_Twb(predicted_pose);
    }
}


void Estimator::update_transform_from_last() {
    if (m_current_frame && m_previous_frame) {
        // Calculate transform from last frame to current frame
        // T_transform = T_last^-1 * T_current
        Eigen::Matrix4f raw_transform = m_previous_frame->get_Twb().inverse() * m_current_frame->get_Twb();
        
        // Apply threshold to translation components (0.01 order and below -> 0)
        Eigen::Vector3f translation = raw_transform.block<3,1>(0,3);
        const float translation_threshold = 0.001f;
        
        for (int i = 0; i < 3; i++) {
            if (std::abs(translation[i]) < translation_threshold) {
                translation[i] = 0.0f;
            }
        }
        
        // Normalize rotation part
        Eigen::Matrix3f rotation = raw_transform.block<3,3>(0,0);
        
        // Check if rotation is close to identity
        Eigen::Matrix3f identity = Eigen::Matrix3f::Identity();
        float rotation_threshold = 0.001f;  // Threshold for rotation similarity
        
        // Calculate Frobenius norm of difference from identity
        Eigen::Matrix3f rotation_diff = rotation - identity;
        float frobenius_norm = rotation_diff.norm();
        
        Eigen::Matrix3f normalized_rotation;
        if (frobenius_norm < rotation_threshold) {
            // Very close to identity - use identity matrix
            normalized_rotation = identity;
        } else {
            // Ensure rotation matrix is properly orthogonal (SVD-based normalization)
            Eigen::JacobiSVD<Eigen::Matrix3f> svd(rotation, Eigen::ComputeFullU | Eigen::ComputeFullV);
            normalized_rotation = svd.matrixU() * svd.matrixV().transpose();
            
            // Ensure proper rotation (det = 1, not -1)
            if (normalized_rotation.determinant() < 0) {
                Eigen::Matrix3f V_corrected = svd.matrixV();
                V_corrected.col(2) *= -1;  // Flip last column
                normalized_rotation = svd.matrixU() * V_corrected.transpose();
            }
        }
        
        // Reconstruct clean transform
        m_transform_from_last = Eigen::Matrix4f::Identity();
        m_transform_from_last.block<3,3>(0,0) = normalized_rotation;
        m_transform_from_last.block<3,1>(0,3) = translation;
    }
}

double lightweight_vio::Estimator::calculate_grid_coverage_with_map_points(std::shared_ptr<Frame> frame) {
    if (!frame || frame->get_feature_count() == 0) {
        return 0.0;
    }
    
    const Config& config = Config::getInstance();
    const int grid_rows = config.m_grid_rows;
    const int grid_cols = config.m_grid_cols;
    const int img_width = config.m_image_width;
    const int img_height = config.m_image_height;
    
    // Initialize grid to track which cells have features with map points
    std::vector<std::vector<bool>> grid_has_map_point(grid_rows, std::vector<bool>(grid_cols, false));
    
    // Check each feature
    const auto& features = frame->get_features();
    for (size_t i = 0; i < features.size(); ++i) {
        const auto& feature = features[i];
        if (!feature || !feature->is_valid()) {
            continue;
        }
        
        // Check if this feature has an associated map point
        auto map_point = frame->get_map_point(i);
        if (!map_point || map_point->is_bad()) {
            continue;
        }
        
        // Calculate grid coordinates
        cv::Point2f pixel_coord = feature->get_pixel_coord();
        
        // Safety check for pixel coordinates
        if (pixel_coord.x < 0 || pixel_coord.x >= img_width || 
            pixel_coord.y < 0 || pixel_coord.y >= img_height) {
            continue;
        }
        
        float cell_width = (float)img_width / grid_cols;
        float cell_height = (float)img_height / grid_rows;
        
        int grid_x = std::min((int)(pixel_coord.x / cell_width), grid_cols - 1);
        int grid_y = std::min((int)(pixel_coord.y / cell_height), grid_rows - 1);
        
        // Additional safety check for grid indices
        if (grid_x >= 0 && grid_x < grid_cols && grid_y >= 0 && grid_y < grid_rows) {
            grid_has_map_point[grid_y][grid_x] = true;
        }
    }
    
    // Count cells with map points
    int cells_with_map_points = 0;
    int total_cells = grid_rows * grid_cols;
    
    for (int row = 0; row < grid_rows; ++row) {
        for (int col = 0; col < grid_cols; ++col) {
            if (grid_has_map_point[row][col]) {
                cells_with_map_points++;
            }
        }
    }
    
    double coverage_ratio = (double)cells_with_map_points / total_cells;
    
    // if (config.m_enable_debug_output) {
    //     // spdlog::debug("Grid coverage: {}/{} cells have features with map points ({:.2f}%)", 
    //     //              cells_with_map_points, total_cells, coverage_ratio * 100.0);
    // }
    
    return coverage_ratio;
}

void Estimator::notify_sliding_window_thread() {
    {
        std::lock_guard<std::mutex> lock(m_keyframes_mutex);
        m_keyframes_updated = true;
    }
    m_keyframes_cv.notify_one();
}

void Estimator::sliding_window_thread_function() {
    spdlog::info("[SW_THREAD] Sliding window optimization thread started");
    
    while (m_sliding_window_thread_running) {
        // Wait for keyframe updates
        std::unique_lock<std::mutex> lock(m_keyframes_mutex);
        m_keyframes_cv.wait(lock, [this] { 
            return !m_sliding_window_thread_running || m_keyframes_updated; 
        });
        
        if (!m_sliding_window_thread_running) {
            break;
        }
        
        if (m_keyframes_updated) {
            m_keyframes_updated = false;
            
            // Copy current keyframes for optimization (thread-safe)
            std::vector<std::shared_ptr<Frame>> keyframes_copy = m_keyframes;
            lock.unlock(); // Release lock early
            
            // Run sliding window bundle adjustment when we have enough keyframes
            if (keyframes_copy.size() >= 2) {
                auto sw_opt_start = std::chrono::high_resolution_clock::now();
                
                auto sw_result = m_sliding_window_optimizer->optimize(keyframes_copy);
                
                auto sw_opt_end = std::chrono::high_resolution_clock::now();
                auto sw_opt_time = std::chrono::duration_cast<std::chrono::microseconds>(sw_opt_end - sw_opt_start).count() / 1000.0;
                
                if (sw_result.success) {
                    spdlog::info("[SW_THREAD] ✅ Optimization successful: {} poses, {} points, {} inliers, {} outliers, cost: {:.2e} -> {:.2e}, time: {:.2f}ms",
                                sw_result.num_poses_optimized, sw_result.num_points_optimized,
                                sw_result.num_inliers, sw_result.num_outliers,
                                sw_result.initial_cost, sw_result.final_cost, sw_opt_time);
                } else {
                    spdlog::warn("[SW_THREAD] ❌ Optimization failed after {:.2f}ms", sw_opt_time);
                }
            } else {
                spdlog::debug("[SW_THREAD] Skipping optimization: only {} keyframes (need ≥2)", keyframes_copy.size());
            }
        }
    }
    
    spdlog::info("[SW_THREAD] Sliding window optimization thread stopped");
}

} // namespace lightweight_vio
