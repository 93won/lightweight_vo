#include "FeatureTracker.h"
#include <algorithm>
#include <iostream>
#include <chrono>

namespace lightweight_vio {

FeatureTracker::FeatureTracker()
    : m_max_features(150)
    , m_quality_level(0.01)
    , m_min_distance(30.0)
    , m_f_threshold(1.0)
    , m_win_size(cv::Size(21, 21))
    , m_max_level(3)
    , m_criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01)
    , m_global_feature_id(0)
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
    if (current_frame->get_feature_count() < m_max_features) {
        set_mask(current_frame);
        extract_new_features(current_frame);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
    
    std::cout << "[TIMING] Total feature tracking: " << total_duration.count() / 1000.0 << " ms | "
              << "Frame " << current_frame->get_frame_id() 
              << " has " << current_frame->get_feature_count() << " features" << std::endl;
}

void FeatureTracker::extract_new_features(std::shared_ptr<Frame> frame) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (frame->get_image().empty()) {
        std::cerr << "Cannot extract features: image is empty" << std::endl;
        return;
    }

    std::vector<cv::Point2f> corners;
    cv::Mat mask = cv::Mat::ones(frame->get_image().size(), CV_8UC1);
    
    // Set mask to avoid existing features
    for (const auto& feature : frame->get_features()) {
        if (feature->is_valid()) {
            cv::circle(mask, feature->get_pixel_coord(), m_min_distance, 0, -1);
        }
    }

    cv::goodFeaturesToTrack(frame->get_image(), corners, 
                           m_max_features - frame->get_feature_count(),
                           m_quality_level, m_min_distance, mask);

    for (const auto& corner : corners) {
        auto feature = std::make_shared<Feature>(m_global_feature_id++, corner);
        frame->add_feature(feature);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "[TIMING] New feature extraction: " << duration.count() / 1000.0 << " ms | "
              << "Extracted " << corners.size() << " new features" << std::endl;
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
    cv::calcOpticalFlowPyrLK(previous_frame->get_image(), current_frame->get_image(),
                            prev_pts, cur_pts, status, err,
                            m_win_size, m_max_level, m_criteria);

    // Create features for current frame based on tracking results
    int tracked_features = 0;
    for (size_t i = 0; i < prev_pts.size(); ++i) {
        if (status[i] && is_in_border(cur_pts[i], current_frame->get_image().size())) {
            auto prev_feature = previous_frame->get_features()[i];
            auto new_feature = std::make_shared<Feature>(
                prev_feature->get_feature_id(),
                cur_pts[i]
            );
            new_feature->set_track_count(prev_feature->get_track_count() + 1);
            current_frame->add_feature(new_feature);
            tracked_features++;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "[TIMING] Optical flow tracking: " << duration.count() / 1000.0 << " ms | "
              << "Tracked " << tracked_features << "/" << prev_pts.size() << " features" << std::endl;
}

void FeatureTracker::reject_outliers_with_fundamental_matrix(std::shared_ptr<Frame> current_frame,
                                                           std::shared_ptr<Frame> previous_frame) {
    if (current_frame->get_feature_count() < 8) {
        return; // Need at least 8 points for fundamental matrix
    }

    std::vector<cv::Point2f> prev_pts, cur_pts;
    std::vector<int> feature_ids;

    // Collect corresponding points
    for (const auto& feature : current_frame->get_features()) {
        auto prev_feature = previous_frame->get_feature(feature->get_feature_id());
        if (prev_feature && prev_feature->is_valid()) {
            prev_pts.push_back(prev_feature->get_pixel_coord());
            cur_pts.push_back(feature->get_pixel_coord());
            feature_ids.push_back(feature->get_feature_id());
        }
    }

    if (prev_pts.size() < 8) {
        return;
    }

    // Find fundamental matrix and inliers
    std::vector<uchar> status;
    cv::findFundamentalMat(prev_pts, cur_pts, cv::FM_RANSAC, m_f_threshold, 0.99, status);

    // Remove outliers
    for (size_t i = 0; i < status.size(); ++i) {
        if (!status[i]) {
            current_frame->remove_feature(feature_ids[i]);
        }
    }

    int outliers_removed = std::count(status.begin(), status.end(), 0);
    std::cout << "Removed " << outliers_removed << " outliers using fundamental matrix" << std::endl;
}

void FeatureTracker::set_mask(std::shared_ptr<Frame> frame) {
    // This is a placeholder for the mask setting function
    // In VINS-Mono, this ensures uniform distribution of features
    // For now, we'll just print a message
    std::cout << "Setting mask for uniform feature distribution (placeholder)" << std::endl;
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
