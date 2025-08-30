#include <module/Estimator.h>
#include <module/PoseOptimizer.h>
#include <module/FeatureTracker.h>
#include <database/Frame.h>
#include <database/MapPoint.h>
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
    , m_current_pose(Eigen::Matrix4f::Identity())
    , m_has_initial_gt_pose(false)
    , m_initial_gt_pose(Eigen::Matrix4f::Identity()) {
    
    // Initialize feature tracker
    m_feature_tracker = std::make_unique<FeatureTracker>();
    
    // Initialize pose optimizer - now uses global Config internally
    m_pose_optimizer = std::make_unique<PoseOptimizer>();
}

Estimator::EstimationResult Estimator::process_frame(const cv::Mat& left_image, const cv::Mat& right_image, long long timestamp) {
    EstimationResult result;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Frame processing starts
    spdlog::info("============================== Frame {} ==============================", m_frame_id_counter);

    // Increment frame counter since last keyframe for every new frame
    m_frames_since_last_keyframe++;

    // Create new stereo frame
    m_current_frame = create_frame(left_image, right_image, timestamp);

    if (!m_current_frame)
    {
        spdlog::error("[Estimator] Failed to create frame!");
        result.success = false;
        return result;
    }

    if (m_previous_frame) {
        // Track features from previous frame using FeatureTracker
        // FeatureTracker now handles both tracking and map point association/creation
        m_feature_tracker->track_features(m_current_frame, m_previous_frame);
        
        result.num_features = m_current_frame->get_feature_count();
        
        // Compute stereo depth for all features
        m_current_frame->compute_stereo_depth();
        
        // Count how many features have associated map points (already done by FeatureTracker)
        int num_tracked_with_map_points = count_features_with_map_points(m_current_frame);
        
        // Log tracking information
        spdlog::info("[TRACKING] {} features tracked, {} with map points", 
                    result.num_features, num_tracked_with_map_points);
        
        if (num_tracked_with_map_points > 0) {
            // DISABLED: Pose optimization temporarily disabled - using GT poses only
            // if (num_tracked_with_map_points >= 5) {
            //     auto opt_result = optimize_pose(m_current_frame);
            //     result.success = opt_result.success;
            //     result.num_inliers = opt_result.num_inliers;
            //     result.num_outliers = opt_result.num_outliers;
            //     
            //     if (opt_result.success) {
            //         m_current_pose = opt_result.optimized_pose;
            //         m_current_frame->set_Twb(m_current_pose);
            //         spdlog::info("[POSE_OPT] Optimization successful: {} inliers, {} outliers", 
            //                     opt_result.num_inliers, opt_result.num_outliers);
            //     } else {
            //         spdlog::warn("[POSE_OPT] Optimization failed");
            //     }
            // } else {
            //     spdlog::warn("[POSE_OPT] Not enough map point associations for optimization: {}", num_tracked_with_map_points);
            // }
            
            // For now, just use the current pose as-is (GT will be applied externally)
            result.success = true;
            result.num_inliers = num_tracked_with_map_points;
            result.num_outliers = 0; 
        } else {
            // No tracking, keep previous pose (already set in create_frame)
            m_current_pose = m_current_frame->get_Twb();
            result.success = false;
        }

        // NOTE: FeatureTracker already handles map point association during tracking
        // No need to call associate_tracked_features_with_map_points() again
        
        
        // Decide whether to create keyframe
        bool is_keyframe = should_create_keyframe(m_current_frame);
        
        // Only create new map points for keyframes to avoid trajectory drift
        if (is_keyframe) {
            int new_map_points = create_new_map_points(m_current_frame);
            result.num_new_map_points = new_map_points;
            spdlog::info("[MAP_POINTS] Created {} new map points by new keyframe insertion", new_map_points);
            create_keyframe(m_current_frame);
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
        m_current_frame->compute_stereo_depth();
        
        // First frame - keep identity pose (already set in create_frame)
        m_current_pose = m_current_frame->get_Twb();
        
        // Increment frame counter (first frame processing)
        m_frames_since_last_keyframe++;
        
        // Create initial map points (first frame is always considered keyframe)
        int initial_map_points = create_initial_map_points(m_current_frame);
        result.num_new_map_points = initial_map_points;
        spdlog::info("[MAP_POINTS] Created {} initial map points", initial_map_points);
        create_keyframe(m_current_frame);
        m_frames_since_last_keyframe = 0;  // Reset after creating first keyframe
        
        // Count features for first frame
        result.num_tracked_features = m_current_frame->get_feature_count();
        result.num_features_with_map_points = count_features_with_map_points(m_current_frame);
        
        result.success = true;
    }
    
    // Update result
    result.pose = m_current_frame->get_Twb();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    result.optimization_time_ms = duration.count() / 1000.0;
    
    // Update state
    m_previous_frame = m_current_frame;
    
    return result;
}

void Estimator::reset() {
    m_current_frame.reset();
    m_previous_frame.reset();
    m_keyframes.clear();
    m_map_points.clear();
    
    m_frame_id_counter = 0;
    m_frames_since_last_keyframe = 0;
    m_current_pose = Eigen::Matrix4f::Identity();
}

Eigen::Matrix4f Estimator::get_current_pose() const {
    return m_current_pose;
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
        global_config.m_baseline,
        global_config.left_dist_coeffs()
    );
    
    // Set initial pose to previous frame's pose (or identity for first frame)
    if (m_previous_frame) {
        // Check if we can get GT pose for current frame
        if (lightweight_vio::EurocUtils::has_ground_truth() && 
            lightweight_vio::EurocUtils::get_matched_count() > m_frame_id_counter) {
            auto gt_pose_opt = lightweight_vio::EurocUtils::get_matched_pose(m_frame_id_counter);
            if (gt_pose_opt.has_value()) {
                frame->set_Twb(gt_pose_opt.value());
                spdlog::debug("[GT_POSE] Applied GT pose for frame {}", m_frame_id_counter);
            } else {
                frame->set_Twb(m_previous_frame->get_Twb());
                spdlog::debug("[PREV_POSE] Used previous frame pose for frame {}", m_frame_id_counter);
            }
        } else {
            frame->set_Twb(m_previous_frame->get_Twb());
            spdlog::debug("[PREV_POSE] Used previous frame pose for frame {}", m_frame_id_counter);
        }
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
                float u_proj = fx * camera_pos.x() / camera_pos.z() + cx;
                float v_proj = fy * camera_pos.y() / camera_pos.z() + cy;
                
                // Get undistorted observed pixel
                cv::Point2f distorted_point(feature->get_u(), feature->get_v());
                cv::Point2f undistorted_point = frame->undistort_point(distorted_point);
                
                // Compute reprojection error
                double error_x = undistorted_point.x - u_proj;
                double error_y = undistorted_point.y - v_proj;
                double error = std::sqrt(error_x * error_x + error_y * error_y);
                
                spdlog::info("[CREATE_MP] #{}: obs=({:.0f},{:.0f}) proj=({:.1f},{:.1f}) err={:.2f}px world=({:.2f},{:.2f},{:.2f})", 
                           num_created, undistorted_point.x, undistorted_point.y, u_proj, v_proj, error,
                           world_pos.x(), world_pos.y(), world_pos.z());
            } else {
                spdlog::warn("[CREATE_MP] #{}: Point behind camera! world=({:.2f},{:.2f},{:.2f})", 
                           num_created, world_pos.x(), world_pos.y(), world_pos.z());
            }
        }
    }
    
    spdlog::info("[MAP_POINTS] Created {} new map points from {} features", num_created, features.size());
    return num_created;
}

int lightweight_vio::Estimator::associate_features_with_map_points(std::shared_ptr<Frame> frame) {
    if (!frame) {
        return 0;
    }
    
    int num_associated = 0;
    const auto& features = frame->get_features();
    
    // Simple association based on proximity (placeholder)
    // In real implementation, you'd use descriptor matching
    for (size_t i = 0; i < features.size() && i < m_map_points.size(); ++i) {
        if (features[i] && features[i]->is_valid() && m_map_points[i] && !m_map_points[i]->is_bad()) {
            frame->set_map_point(i, m_map_points[i]);
            num_associated++;
        }
    }
    
    return num_associated;
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
        spdlog::debug("NEW Map Point {}: world_pos=({:.3f}, {:.3f}, {:.3f})", 
                     m_map_points.size() - 1, world_pos.x(), world_pos.y(), world_pos.z());
        
        // Verify reprojection in current frame
        Eigen::Matrix4f T_wc = T_wb * T_bc;  // World to camera
        Eigen::Matrix4f T_cw = T_wc.inverse(); // Camera to world (for projection)
        
        // Project back to image
        Eigen::Vector4f world_homogeneous(world_pos.x(), world_pos.y(), world_pos.z(), 1.0);
        Eigen::Vector4f camera_projected = T_cw * world_homogeneous;
        
        if (camera_projected.z() > 0) {  // Valid projection
            auto feature = frame->get_feature(i);
            if (feature) {
                cv::Point2f observed_pt = feature->get_pixel_coord();
                
                // Get camera intrinsics from frame
                double fx, fy, cx, cy;
                frame->get_camera_intrinsics(fx, fy, cx, cy);
                
                float projected_x = (fx * camera_projected.x() / camera_projected.z()) + cx;
                float projected_y = (fy * camera_projected.y() / camera_projected.z()) + cy;
                
                float reprojection_error = sqrt(pow(observed_pt.x - projected_x, 2) + pow(observed_pt.y - projected_y, 2));
                
                spdlog::debug("  Reprojection: observed=({:.2f}, {:.2f}), projected=({:.2f}, {:.2f}), error={:.3f}px", 
                             observed_pt.x, observed_pt.y, projected_x, projected_y, reprojection_error);
            }
        }
        
        num_created++;
    }
    
    return num_created;
}

int lightweight_vio::Estimator::associate_tracked_features_with_map_points(std::shared_ptr<Frame> frame) {
    if (!frame || !m_previous_frame) {
        return 0;
    }
    
    int num_associated = 0;
    const auto& features = frame->get_features();
    
    // For each feature in current frame, check if it was tracked from previous frame
    // and if the previous frame feature had a map point
    for (size_t i = 0; i < features.size(); ++i) {
        auto feature = features[i];
        if (!feature || !feature->is_valid()) {
            continue;
        }
        
        // Skip if this feature already has a map point
        if (frame->has_map_point(i)) {
            continue;
        }
        
        // Get feature track ID to find corresponding feature in previous frame
        int track_id = feature->get_tracked_feature_id();
        
        // Find corresponding feature in previous frame by track ID
        const auto& prev_features = m_previous_frame->get_features();
        for (size_t j = 0; j < prev_features.size(); ++j) {
            auto prev_feature = prev_features[j];
            if (prev_feature && prev_feature->is_valid() && 
                prev_feature->get_tracked_feature_id() == track_id && track_id >= 0) {
                
                // Check if previous feature has associated map point
                auto prev_map_point = m_previous_frame->get_map_point(j);
                if (prev_map_point && !prev_map_point->is_bad()) {
                    // Associate with existing map point
                    frame->set_map_point(i, prev_map_point);
                    prev_map_point->add_observation(frame, i);
                    num_associated++;
                    break;
                }
            }
        }
    }
    
    return num_associated;
}

bool lightweight_vio::Estimator::should_create_keyframe(std::shared_ptr<Frame> frame) {

    if (!frame) {
        return false;
    }
    
    // Simple keyframe creation policy
    if (m_keyframes.empty()) {
        return true;  // First frame is always a keyframe
    }
    
    if (m_frames_since_last_keyframe >= Config::getInstance().m_keyframe_interval) {
        return true;
    }
    
    // Could add more sophisticated criteria:
    // Could add more sophisticated criteria:
    // - Translation/rotation distance from last keyframe
    // - Number of inliers
    // - Feature distribution
    
    return false;
}

void lightweight_vio::Estimator::create_keyframe(std::shared_ptr<Frame> frame) {
    if (!frame) {
        return;
    }
    
    frame->set_keyframe(true);
    m_keyframes.push_back(frame);
    
    spdlog::info("[KEYFRAME] Created keyframe {} with {} features", 
                frame->get_frame_id(), frame->get_feature_count());
}

OptimizationResult lightweight_vio::Estimator::optimize_pose(std::shared_ptr<Frame> frame) {
    // Create pose optimizer - uses global Config internally
    PoseOptimizer optimizer;
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

void lightweight_vio::Estimator::update_map_points(std::shared_ptr<Frame> frame) {
    if (!frame) {
        return;
    }
    
    // Update map point observations and remove bad points
    const auto& map_points = frame->get_map_points();
    const auto& outlier_flags = frame->get_outlier_flags();
    
    for (size_t i = 0; i < map_points.size() && i < outlier_flags.size(); ++i) {
        auto mp = map_points[i];
        if (mp) {
            if (outlier_flags[i]) {
                // Mark as bad or remove
                mp->set_bad();
            } else {
                // Update observations, quality, etc.
                mp->add_observation(frame, i);
            }
        }
    }
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

        std::cout<<gt_pose<<"\n";
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
    spdlog::debug("[REPROJ_DEBUG] Camera params: fx={:.2f}, fy={:.2f}, cx={:.2f}, cy={:.2f}", fx, fy, cx, cy);
    
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
        
        // Project to image plane
        float u_proj = fx * camera_pos.x() / camera_pos.z() + cx;
        float v_proj = fy * camera_pos.y() / camera_pos.z() + cy;
        
        // Get undistorted observed pixel
        cv::Point2f distorted_point(feature->get_u(), feature->get_v());
        cv::Point2f undistorted_point = frame->undistort_point(distorted_point);
        
        // Compute reprojection error
        double error_x = undistorted_point.x - u_proj;
        double error_y = undistorted_point.y - v_proj;
        double error = std::sqrt(error_x * error_x + error_y * error_y);
        
        reprojection_errors.push_back(error);
        valid_projections++;
        
        // Print core info for all features in one line
        spdlog::info("[REPROJ] F{}: obs=({:.0f},{:.0f}) proj=({:.1f},{:.1f}) err={:.2f}px world=({:.2f},{:.2f},{:.2f})", 
                     i, undistorted_point.x, undistorted_point.y, u_proj, v_proj, error,
                     world_pos.x(), world_pos.y(), world_pos.z());
    }
    
    if (valid_projections > 0) {
        // Compute statistics
        std::sort(reprojection_errors.begin(), reprojection_errors.end());
        
        double mean_error = std::accumulate(reprojection_errors.begin(), reprojection_errors.end(), 0.0) / reprojection_errors.size();
        double median_error = reprojection_errors[reprojection_errors.size() / 2];
        double min_error = reprojection_errors.front();
        double max_error = reprojection_errors.back();
        
        // Count outliers (error > 5 pixels)
        int outliers = std::count_if(reprojection_errors.begin(), reprojection_errors.end(), 
                                   [](double error) { return error > 5.0; });
        
        spdlog::info("[REPROJ_ERROR] Frame {}: {}/{} valid, mean={:.2f}px, median={:.2f}px, min={:.2f}px, max={:.2f}px, outliers={}", 
                    frame->get_frame_id(), valid_projections, features.size(), 
                    mean_error, median_error, min_error, max_error, outliers);
        
        if (behind_camera > 0) {
            spdlog::warn("[REPROJ_ERROR] {} points behind camera", behind_camera);
        }
    } else {
        spdlog::warn("[REPROJ_ERROR] Frame {}: No valid projections", frame->get_frame_id());
    }
}

} // namespace lightweight_vio
