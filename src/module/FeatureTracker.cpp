#include "FeatureTracker.h"
#include <algorithm>
#include <iostream>
#include <chrono>

namespace lightweight_vio {

FeatureTracker::FeatureTracker()
    : m_global_feature_id(0)
{
}

void FeatureTracker::track_features(std::shared_ptr<Frame> current_frame, 
                                   std::shared_ptr<Frame> previous_frame) {
    auto total_start = std::chrono::high_resolution_clock::now();
    
    if (!current_frame) {
        std::cerr << "Current frame is null" << std::endl;
        return;
    }

    if (previous_frame) {
        // Track existing features
        optical_flow_tracking(current_frame, previous_frame);
        
        // Reject outliers using fundamental matrix
        reject_outliers_with_fundamental_matrix(current_frame, previous_frame);
        
        // Update track counts
        update_feature_track_count(current_frame);
    }

    // Extract new features if needed
    if (current_frame->get_feature_count() < m_config.getMaxFeatures()) {
        set_mask(current_frame);
        extract_new_features(current_frame);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
    
    if (m_config.isTimingEnabled()) {
        std::cout << "[TIMING] Total feature tracking: " << total_duration.count() / 1000.0 << " ms | "
                  << "Frame " << current_frame->get_frame_id() 
                  << " has " << current_frame->get_feature_count() << " features" << std::endl;
    }
}

void FeatureTracker::extract_new_features(std::shared_ptr<Frame> frame) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (frame->get_image().empty()) {
        std::cerr << "Cannot extract features: image is empty" << std::endl;
        return;
    }

    std::vector<cv::Point2f> corners;
    
    // Use the mask created by set_mask function
    cv::Mat mask_to_use = m_mask.empty() ? cv::Mat() : m_mask;

    int features_needed = Config::getInstance().getMaxFeatures() - frame->get_feature_count();
    
    if (features_needed > 0) {
        cv::goodFeaturesToTrack(frame->get_image(), corners, 
                               features_needed,
                               Config::getInstance().getQualityLevel(), 
                               Config::getInstance().getMinDistance(), 
                               mask_to_use);

        for (const auto& corner : corners) {
            auto feature = std::make_shared<Feature>(m_global_feature_id++, corner);
            frame->add_feature(feature);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    const Config& config = Config::getInstance();
    if (config.isTimingEnabled()) {
        std::cout << "[TIMING] New feature extraction: " << duration.count() / 1000.0 << " ms | "
                  << "Extracted " << corners.size() << " new features" << std::endl;
    }
}

void FeatureTracker::optical_flow_tracking(std::shared_ptr<Frame> current_frame,
                                          std::shared_ptr<Frame> previous_frame) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (previous_frame->get_feature_count() == 0) {
        return;
    }

    // Extract points from previous frame features
    std::vector<cv::Point2f> prev_pts = extract_points_from_features(previous_frame->get_features());
    std::vector<cv::Point2f> cur_pts;
    std::vector<uchar> status;
    std::vector<float> err;

    // Perform optical flow tracking
    int window_size = Config::getInstance().getWindowSize();
    cv::calcOpticalFlowPyrLK(previous_frame->get_image(), current_frame->get_image(),
                            prev_pts, cur_pts, status, err,
                            cv::Size(window_size, window_size), 
                            Config::getInstance().getMaxLevel(), 
                            Config::getInstance().getTermCriteria());

    // Create features for current frame based on tracking results
    int tracked_features = 0;
    for (size_t i = 0; i < prev_pts.size(); ++i) {
        if (status[i] && is_in_border(cur_pts[i], current_frame->get_image().size())) {
            // Additional checks for tracking quality
            float dx = cur_pts[i].x - prev_pts[i].x;
            float dy = cur_pts[i].y - prev_pts[i].y;
            float movement = std::sqrt(dx * dx + dy * dy);
            
            // Reject if error is too high or movement is too large
            if (err[i] < Config::getInstance().getErrorThreshold() && 
                movement < Config::getInstance().getMaxMovementDistance()) {
                auto prev_feature = previous_frame->get_features()[i];
                auto new_feature = std::make_shared<Feature>(
                    prev_feature->get_feature_id(),
                    cur_pts[i]
                );
                
                // Update velocity
                Eigen::Vector2f velocity(dx, dy);
                new_feature->set_velocity(velocity);
                new_feature->set_track_count(prev_feature->get_track_count() + 1);
                current_frame->add_feature(new_feature);
                tracked_features++;
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    if (m_config.isTimingEnabled()) {
        std::cout << "[TIMING] Optical flow tracking: " << duration.count() / 1000.0 << " ms | "
                  << "Tracked " << tracked_features << "/" << prev_pts.size() << " features" << std::endl;
    }
}

void FeatureTracker::reject_outliers_with_fundamental_matrix(std::shared_ptr<Frame> current_frame,
                                                           std::shared_ptr<Frame> previous_frame) {
    if (current_frame->get_feature_count() < 8) {
        return; // Need at least 8 points for fundamental matrix
    }

    std::vector<cv::Point2f> prev_pts, cur_pts;
    std::vector<int> feature_ids;
    std::vector<std::shared_ptr<Feature>> features_to_check;

    // Collect corresponding points
    for (const auto& feature : current_frame->get_features()) {
        auto prev_feature = previous_frame->get_feature(feature->get_feature_id());
        if (prev_feature && prev_feature->is_valid()) {
            prev_pts.push_back(prev_feature->get_pixel_coord());
            cur_pts.push_back(feature->get_pixel_coord());
            feature_ids.push_back(feature->get_feature_id());
            features_to_check.push_back(feature);
        }
    }

    if (prev_pts.size() < 8) {
        return;
    }

    std::vector<int> outlier_features;

    // Step 1: Movement distance check (reject features that moved too far)
    for (size_t i = 0; i < prev_pts.size(); ++i) {
        float dx = cur_pts[i].x - prev_pts[i].x;
        float dy = cur_pts[i].y - prev_pts[i].y;
        float movement = std::sqrt(dx * dx + dy * dy);
        
        // Reject if movement is too large (likely tracking error)
        if (movement > Config::getInstance().getMaxMovementDistance()) {
            outlier_features.push_back(feature_ids[i]);
        }
    }

    // Step 2: Fundamental matrix RANSAC
    std::vector<uchar> status;
    if (prev_pts.size() >= 8) {
        cv::findFundamentalMat(prev_pts, cur_pts, cv::FM_RANSAC, 
                              Config::getInstance().getFundamentalThreshold(), 0.99, status);
        
        // Mark fundamental matrix outliers
        for (size_t i = 0; i < status.size(); ++i) {
            if (!status[i]) {
                // Check if not already marked as outlier
                if (std::find(outlier_features.begin(), outlier_features.end(), feature_ids[i]) == outlier_features.end()) {
                    outlier_features.push_back(feature_ids[i]);
                }
            }
        }
    }

    // Step 3: Velocity consistency check (sudden direction changes)
    for (size_t i = 0; i < features_to_check.size(); ++i) {
        auto feature = features_to_check[i];
        if (feature->get_track_count() > 1) {  // Need at least 2 frames of history
            Eigen::Vector2f curr_vel = feature->get_velocity();
            float dx = cur_pts[i].x - prev_pts[i].x;
            float dy = cur_pts[i].y - prev_pts[i].y;
            
            // Check for sudden velocity change
            float vel_diff_x = std::abs(dx - curr_vel.x());
            float vel_diff_y = std::abs(dy - curr_vel.y());
            
            if (vel_diff_x > Config::getInstance().getMaxVelocityChange() || 
                vel_diff_y > Config::getInstance().getMaxVelocityChange()) {
                if (std::find(outlier_features.begin(), outlier_features.end(), feature_ids[i]) == outlier_features.end()) {
                    outlier_features.push_back(feature_ids[i]);
                }
            }
        }
    }

    // Remove all outliers
    for (int feature_id : outlier_features) {
        current_frame->remove_feature(feature_id);
    }

    if (!outlier_features.empty() && m_config.isDebugOutputEnabled()) {
        std::cout << "Removed " << outlier_features.size() << " outliers (movement + fundamental matrix + velocity)" << std::endl;
    }
}

void FeatureTracker::set_mask(std::shared_ptr<Frame> frame) {
    if (frame->get_image().empty()) {
        std::cerr << "Cannot set mask: image is empty" << std::endl;
        return;
    }
    
    // Create mask initialized to 255 (valid areas for feature detection)
    m_mask = cv::Mat(frame->get_image().size(), CV_8UC1, cv::Scalar(255));
    
    const Config& config = Config::getInstance();
    int min_distance = static_cast<int>(config.getMinDistance());
    int border_size = config.getBorderSize();
    
    // Set border regions to 0 (invalid for feature detection)
    if (border_size > 0) {
        m_mask(cv::Rect(0, 0, frame->get_image().cols, border_size)) = 0;                    // Top
        m_mask(cv::Rect(0, frame->get_image().rows - border_size, frame->get_image().cols, border_size)) = 0; // Bottom
        m_mask(cv::Rect(0, 0, border_size, frame->get_image().rows)) = 0;                    // Left
        m_mask(cv::Rect(frame->get_image().cols - border_size, 0, border_size, frame->get_image().rows)) = 0; // Right
    }
    
    // For each existing feature, create a circular mask around it
    for (const auto& feature : frame->get_features()) {
        if (feature->is_valid()) {
            cv::Point2f pt = feature->get_pixel_coord();
            
            // Check if feature is within image bounds
            if (pt.x >= 0 && pt.y >= 0 && 
                pt.x < frame->get_image().cols && pt.y < frame->get_image().rows) {
                
                // Create circular mask around existing feature
                cv::circle(m_mask, pt, min_distance, cv::Scalar(0), -1);
            }
        }
    }
    
    if (config.isDebugOutputEnabled()) {
        int valid_pixels = cv::countNonZero(m_mask);
        int total_pixels = m_mask.rows * m_mask.cols;
        std::cout << "[DEBUG] Mask set: " << valid_pixels << "/" << total_pixels 
                  << " pixels available for new features (" 
                  << (100.0 * valid_pixels / total_pixels) << "%)" << std::endl;
    }
}

void FeatureTracker::update_feature_track_count(std::shared_ptr<Frame> frame) {
    for (auto& feature : frame->get_features()) {
        if (feature->is_valid()) {
            feature->set_track_count(feature->get_track_count() + 1);
        }
    }
}

std::vector<cv::Point2f> FeatureTracker::extract_points_from_features(
    const std::vector<std::shared_ptr<Feature>>& features) {
    std::vector<cv::Point2f> points;
    for (const auto& feature : features) {
        if (feature->is_valid()) {
            points.push_back(feature->get_pixel_coord());
        }
    }
    return points;
}

void FeatureTracker::update_features_with_points(
    std::vector<std::shared_ptr<Feature>>& features,
    const std::vector<cv::Point2f>& points,
    const std::vector<uchar>& status) {
    
    size_t point_idx = 0;
    for (auto& feature : features) {
        if (feature->is_valid() && point_idx < points.size()) {
            if (status[point_idx]) {
                feature->set_pixel_coord(points[point_idx]);
            } else {
                feature->set_valid(false);
            }
            point_idx++;
        }
    }
}

bool FeatureTracker::is_in_border(const cv::Point2f& point, const cv::Size& img_size, int border_size) const {
    int img_x = cvRound(point.x);
    int img_y = cvRound(point.y);
    return border_size <= img_x && img_x < img_size.width - border_size && 
           border_size <= img_y && img_y < img_size.height - border_size;
}

} // namespace lightweight_vio
