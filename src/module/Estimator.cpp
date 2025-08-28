#include "Estimator.h"
#include "PoseOptimizer.h"
#include <chrono>
#include <iostream>

namespace lightweight_vio {

Estimator::Estimator(const Config& config)
    : m_config(config)
    , m_frame_id_counter(0)
    , m_frames_since_last_keyframe(0)
    , m_current_pose(Eigen::Matrix4f::Identity()) {
    
    // Initialize feature tracker
    m_feature_tracker = std::make_unique<FeatureTracker>();
    
    // Initialize pose optimizer
    m_pose_optimizer = std::make_unique<PoseOptimizer>(m_config.pose_optimizer_config);
}

Estimator::EstimationResult Estimator::process_frame(const cv::Mat& left_image, const cv::Mat& right_image, long long timestamp) {
    EstimationResult result;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Create new stereo frame
    m_current_frame = create_frame(left_image, right_image, timestamp);
    
    if (!m_current_frame) {
        std::cerr << "[ERROR] Failed to create frame!" << std::endl;
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
        
        if (num_tracked_with_map_points > 0) {
            // Perform pose optimization if we have enough associations
            if (m_config.enable_pose_optimization && num_tracked_with_map_points >= 5) {
                auto opt_result = optimize_pose(m_current_frame);
                result.success = opt_result.success;
                result.num_inliers = opt_result.num_inliers;
                result.num_outliers = opt_result.num_outliers;
                
                if (opt_result.success) {
                    m_current_pose = opt_result.optimized_pose;
                    m_current_frame->set_Twb(m_current_pose);
                }
            } else {
                // Use previous pose as initial guess
                m_current_frame->set_Twb(m_current_pose);
                result.success = true;
            }
        } else {
            // No tracking, use previous pose
            m_current_frame->set_Twb(m_current_pose);
            result.success = false;
        }
    } else {
        // First frame - extract features using FeatureTracker
        m_feature_tracker->track_features(m_current_frame, nullptr);
        
        result.num_features = m_current_frame->get_feature_count();
        
        // Compute stereo depth for all features
        m_current_frame->compute_stereo_depth();
        
        // First frame - initialize at origin
        m_current_pose = Eigen::Matrix4f::Identity();
        m_current_frame->set_Twb(m_current_pose);
        
        // Create initial map points
        int initial_map_points = create_initial_map_points(m_current_frame);
        
        result.success = true;
    }
    
    // Decide whether to create keyframe
    if (should_create_keyframe(m_current_frame)) {
        create_keyframe(m_current_frame);
        m_frames_since_last_keyframe = 0;
    } else {
        m_frames_since_last_keyframe++;
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
    auto frame = std::make_shared<Frame>(
        timestamp, 
        m_frame_id_counter++,
        gray_left, gray_right,
        m_config.fx, m_config.fy, m_config.cx, m_config.cy,
        m_config.baseline,
        cv::Mat(m_config.distortion_coeffs)
    );
    
    return frame;
}



int lightweight_vio::Estimator::create_initial_map_points(std::shared_ptr<Frame> frame) {
    if (!frame) {
        return 0;
    }
    
    int num_created = 0;
    const auto& features = frame->get_features();
    
    // Create map points at arbitrary depth for initialization
    // In real implementation, you'd use stereo or structure from motion
    for (size_t i = 0; i < features.size(); ++i) {
        auto feature = features[i];
        if (feature && feature->is_valid() && frame->has_depth(i)) {
            // Use stereo depth for map point creation
            double depth = frame->get_depth(i);
            
            // Unproject to 3D using camera parameters and stereo depth
            double fx, fy, cx, cy;
            frame->get_camera_intrinsics(fx, fy, cx, cy);
            
            cv::Point2f undistorted = frame->undistort_point(cv::Point2f(feature->get_u(), feature->get_v()));
            
            double x = (undistorted.x - cx) * depth / fx;
            double y = (undistorted.y - cy) * depth / fy;
            double z = depth;
            
            // Transform to world coordinates using frame pose
            Eigen::Matrix4f Twb = frame->get_Twb();
            Eigen::Vector4f camera_point(x, y, z, 1.0);
            Eigen::Vector4f world_point = Twb * camera_point;
            
            Eigen::Vector3f world_pos = world_point.head<3>();
            
            auto map_point = std::make_shared<MapPoint>(world_pos);
            m_map_points.push_back(map_point);
            
            // Associate with frame
            frame->set_map_point(i, map_point);
            num_created++;
        }
    }
    
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

bool lightweight_vio::Estimator::should_create_keyframe(std::shared_ptr<Frame> frame) {
    if (!frame) {
        return false;
    }
    
    // Simple keyframe creation policy
    if (m_keyframes.empty()) {
        return true;  // First frame is always a keyframe
    }
    
    if (m_frames_since_last_keyframe >= m_config.keyframe_interval) {
        return true;
    }
    
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
    
    if (m_config.pose_optimizer_config.print_summary) {
        std::cout << "Created keyframe " << frame->get_frame_id() 
                  << " (total: " << m_keyframes.size() << ")" << std::endl;
    }
}

OptimizationResult lightweight_vio::Estimator::optimize_pose(std::shared_ptr<Frame> frame) {
    // TODO: Implement pose optimization using PoseOptimizer
    PoseOptimizer optimizer(m_config.pose_optimizer_config);
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

} // namespace lightweight_vio
