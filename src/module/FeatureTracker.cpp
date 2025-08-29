#include <module/FeatureTracker.h>
#include <database/Frame.h>
#include <database/Feature.h>
#include <database/MapPoint.h>
#include <spdlog/spdlog.h>
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <chrono>

namespace lightweight_vio {

FeatureTracker::FeatureTracker()
{
}

void FeatureTracker::track_features(std::shared_ptr<Frame> current_frame, 
                                   std::shared_ptr<Frame> previous_frame) {
    auto total_start = std::chrono::high_resolution_clock::now();
    
    if (!current_frame) {
        std::cerr << "Current frame is null" << std::endl;
        return;
    }

    int tracked_features = 0;
    int new_map_points_from_tracking = 0;
    int new_extracted_features = 0;
    int new_map_points_from_extraction = 0;

    auto tracking_time = 0.0;
    auto outlier_rejection_time = 0.0;
    auto feature_extraction_time = 0.0;
    auto mask_creation_time = 0.0;

    if (previous_frame) {
        // Track existing features
        auto tracking_start = std::chrono::high_resolution_clock::now();
        auto tracking_stats = optical_flow_tracking(current_frame, previous_frame);
        auto tracking_end = std::chrono::high_resolution_clock::now();
        tracking_time = std::chrono::duration_cast<std::chrono::microseconds>(tracking_end - tracking_start).count() / 1000.0;
        
        tracked_features = tracking_stats.first;
        new_map_points_from_tracking = tracking_stats.second;
        
        // Reject outliers using comprehensive filtering
        auto outlier_start = std::chrono::high_resolution_clock::now();
        reject_outliers(current_frame, previous_frame);
        auto outlier_end = std::chrono::high_resolution_clock::now();
        outlier_rejection_time = std::chrono::duration_cast<std::chrono::microseconds>(outlier_end - outlier_start).count() / 1000.0;
        
        // Update track counts
        update_feature_track_count(current_frame);
    }

    // Extract new features if needed
    if (current_frame->get_feature_count() < m_config.m_max_features) {
        auto mask_start = std::chrono::high_resolution_clock::now();
        set_mask(current_frame);
        auto mask_end = std::chrono::high_resolution_clock::now();
        mask_creation_time = std::chrono::duration_cast<std::chrono::microseconds>(mask_end - mask_start).count() / 1000.0;
        
        auto extraction_start = std::chrono::high_resolution_clock::now();
        auto extraction_stats = extract_new_features(current_frame);
        auto extraction_end = std::chrono::high_resolution_clock::now();
        feature_extraction_time = std::chrono::duration_cast<std::chrono::microseconds>(extraction_end - extraction_start).count() / 1000.0;
        
        new_extracted_features = extraction_stats.first;
        new_map_points_from_extraction = extraction_stats.second;
    }

    // Now do batch stereo matching and map point creation for all features without map points
    auto batch_stereo_start = std::chrono::high_resolution_clock::now();
    // Temporarily disable automatic map point creation in FeatureTracker
    // Let Estimator handle map point creation instead
    // int batch_stereo_matches = batch_stereo_matching_and_map_point_creation(current_frame);
    int batch_stereo_matches = 0;
    auto batch_stereo_end = std::chrono::high_resolution_clock::now();
    auto batch_stereo_time = std::chrono::duration_cast<std::chrono::microseconds>(batch_stereo_end - batch_stereo_start).count() / 1000.0;
    
    // Update total new map points (from tracking + extraction + batch stereo)
    new_map_points_from_tracking += batch_stereo_matches;
    int total_new_map_points = new_map_points_from_tracking + new_map_points_from_extraction;

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
    
    // Count final valid map points
    int final_valid_map_points = 0;
    
    for (size_t i = 0; i < current_frame->get_map_points().size(); ++i) {
        auto mp = current_frame->get_map_point(i);
        if (mp && !mp->is_bad()) {
            final_valid_map_points++;
        }
    }
    
    // Single comprehensive log with timing breakdown
    auto total_time = total_duration.count() / 1000.0;
    
   
    
    // Timing output removed for cleaner logs
}

std::pair<int, int> FeatureTracker::extract_new_features(std::shared_ptr<Frame> frame) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (frame->get_image().empty()) {
        std::cerr << "Cannot extract features: image is empty" << std::endl;
        return {0, 0};
    }

    std::vector<cv::Point2f> corners;
    
    // Use the mask created by set_mask function
    cv::Mat mask_to_use = m_mask.empty() ? cv::Mat() : m_mask;

    int features_needed = Config::getInstance().m_max_features - frame->get_feature_count();
    int new_map_points_created = 0;
    
    if (features_needed > 0) {
        auto detection_start = std::chrono::high_resolution_clock::now();
        cv::goodFeaturesToTrack(frame->get_image(), corners, 
                               features_needed,
                               Config::getInstance().m_quality_level, 
                               Config::getInstance().m_min_distance, 
                               mask_to_use);
        auto detection_end = std::chrono::high_resolution_clock::now();
        auto detection_time = std::chrono::duration_cast<std::chrono::microseconds>(detection_end - detection_start).count() / 1000.0;

        auto feature_processing_start = std::chrono::high_resolution_clock::now();
        
        // Create all features without map points - batch processing will handle stereo matching later
        int next_feature_id = frame->get_feature_count();  // Start from current count
        for (const auto& corner : corners) {
            auto feature = std::make_shared<Feature>(next_feature_id++, corner);
            frame->add_feature(feature);
        }
        
        // Return feature count and 0 for immediate map points (will be created in batch later)
        return std::make_pair(corners.size(), 0);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    return {static_cast<int>(corners.size()), 0};
}

std::pair<int, int> FeatureTracker::optical_flow_tracking(std::shared_ptr<Frame> current_frame,
                                          std::shared_ptr<Frame> previous_frame) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (previous_frame->get_feature_count() == 0) {
        return {0, 0};
    }

    // Extract points from previous frame features and keep track of original indices
    auto point_extraction_start = std::chrono::high_resolution_clock::now();
    std::vector<cv::Point2f> prev_pts;
    std::vector<int> feature_indices;  // Track original feature indices
    
    for (size_t idx = 0; idx < previous_frame->get_features().size(); ++idx) {
        const auto& feature = previous_frame->get_features()[idx];
        if (feature->is_valid()) {
            prev_pts.push_back(feature->get_pixel_coord());
            feature_indices.push_back(idx);  // Store the original index
        }
    }
    auto point_extraction_end = std::chrono::high_resolution_clock::now();
    auto point_extraction_time = std::chrono::duration_cast<std::chrono::microseconds>(point_extraction_end - point_extraction_start).count() / 1000.0;
    
    std::vector<cv::Point2f> cur_pts;
    std::vector<uchar> status;
    std::vector<float> err;

    // Perform optical flow tracking
    auto flow_start = std::chrono::high_resolution_clock::now();
    int window_size = Config::getInstance().m_window_size;
    
    // Use OpenCV implementation (most stable and optimized)
    cv::calcOpticalFlowPyrLK(previous_frame->get_image(), current_frame->get_image(),
                            prev_pts, cur_pts, status, err,
                            cv::Size(window_size, window_size), 
                            Config::getInstance().m_max_level, 
                            Config::getInstance().term_criteria());
    
    auto flow_end = std::chrono::high_resolution_clock::now();
    auto flow_time = std::chrono::duration_cast<std::chrono::microseconds>(flow_end - flow_start).count() / 1000.0;

    // Create features for current frame based on tracking results
    auto feature_creation_start = std::chrono::high_resolution_clock::now();
    int tracked_features = 0;
    int associated_map_points = 0;
    int new_map_points_created = 0;
    int next_feature_id = current_frame->get_feature_count();  // Start from current count
    
    for (size_t i = 0; i < prev_pts.size(); ++i) {
        if (status[i] && is_in_border(cur_pts[i], current_frame->get_image().size())) {
            // Basic checks for tracking quality (relaxed thresholds - main filtering done later)
            float dx = cur_pts[i].x - prev_pts[i].x;
            float dy = cur_pts[i].y - prev_pts[i].y;
            
            // Only reject extremely bad tracking results here, main outlier rejection done later
            if (err[i] < Config::getInstance().m_error_threshold * 2.0) {  // Relaxed error threshold
                // Get the original feature index
                int original_feature_idx = feature_indices[i];
                auto prev_feature = previous_frame->get_features()[original_feature_idx];
                
                // Create new feature with sequential frame-local ID
                auto new_feature = std::make_shared<Feature>(
                    next_feature_id++,  // Use sequential ID
                    cur_pts[i]
                );
                
                // Set the tracked feature ID to maintain tracking relationship
                new_feature->set_tracked_feature_id(prev_feature->get_feature_id());
                
                // Update velocity
                Eigen::Vector2f velocity(dx, dy);
                new_feature->set_velocity(velocity);
                
                // Propagate outlier flag from previous frame
                bool was_outlier = previous_frame->get_outlier_flag(original_feature_idx);
                current_frame->add_feature(new_feature);
                current_frame->set_outlier_flag(current_frame->get_feature_count() - 1, was_outlier);
                new_feature->set_track_count(prev_feature->get_track_count() + 1);
                tracked_features++;
                
                // Check if previous feature has associated map point
                // Use the original feature index to get the correct map point
                auto prev_map_point = previous_frame->get_map_point(original_feature_idx);
                
                if (prev_map_point && !prev_map_point->is_bad()) {
                    // Debug: Check if this association looks suspicious
                    cv::Point2f current_pt = cur_pts[i];  // Use cur_pts[i] instead of tracked_features[j]
                    cv::Point2f prev_pt = prev_pts[i];    // Use prev_pts[i] instead of previous_features
                    float tracking_distance = cv::norm(current_pt - prev_pt);
                    
                    // Associate with existing map point
                    current_frame->set_map_point(current_frame->get_feature_count() - 1, prev_map_point);
                    prev_map_point->add_observation(current_frame, current_frame->get_feature_count() - 1);
                    associated_map_points++;
                    
                   
                } 
                
            }
        }
    }
    auto feature_creation_end = std::chrono::high_resolution_clock::now();
    auto feature_creation_time = std::chrono::duration_cast<std::chrono::microseconds>(feature_creation_end - feature_creation_start).count() / 1000.0;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
    
    // Log detailed timing for slow optical flow operations
    if (total_time > 5.0) {
        std::cout << "[FLOW DEBUG] Total: " << total_time << "ms | "
                  << "Point extraction: " << point_extraction_time << "ms, "
                  << "Optical flow: " << flow_time << "ms, "
                  << "Feature creation: " << feature_creation_time << "ms | "
                  << "Points: " << prev_pts.size() << " -> " << tracked_features << std::endl;
    }
    
    return {tracked_features, new_map_points_created};
}

void FeatureTracker::reject_outliers(std::shared_ptr<Frame> current_frame,
                                     std::shared_ptr<Frame> previous_frame) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (current_frame->get_feature_count() < 5) {
        return; // Need at least 5 points for essential matrix
    }

    auto point_collection_start = std::chrono::high_resolution_clock::now();
    std::vector<cv::Point2f> prev_pts, cur_pts;
    std::vector<int> feature_ids;
    std::vector<std::shared_ptr<Feature>> features_to_check;

    // Collect corresponding points
    for (const auto& feature : current_frame->get_features()) {
        // Use tracked_feature_id to find the corresponding previous feature
        if (feature->has_tracked_feature()) {
            int tracked_id = feature->get_tracked_feature_id();
            // Find previous feature index using the proper mapping
            int prev_feature_idx = previous_frame->get_feature_index(tracked_id);
            if (prev_feature_idx >= 0) {
                auto prev_feature = previous_frame->get_features()[prev_feature_idx];
                if (prev_feature && prev_feature->is_valid()) {
                    prev_pts.push_back(prev_feature->get_pixel_coord());
                    cur_pts.push_back(feature->get_pixel_coord());
                    feature_ids.push_back(feature->get_feature_id());
                    features_to_check.push_back(feature);
                }
            }
        }
    }
    auto point_collection_end = std::chrono::high_resolution_clock::now();
    auto point_collection_time = std::chrono::duration_cast<std::chrono::microseconds>(point_collection_end - point_collection_start).count() / 1000.0;

    if (prev_pts.size() < 5) {
        return;
    }

    std::vector<int> outlier_features;
    std::vector<int> movement_outliers;
    std::vector<int> ransac_outliers;
    std::vector<int> velocity_outliers;

    // Step 1: Movement distance check (main movement filtering done here)
    for (size_t i = 0; i < prev_pts.size(); ++i) {
        float dx = cur_pts[i].x - prev_pts[i].x;
        float dy = cur_pts[i].y - prev_pts[i].y;
        float movement = std::sqrt(dx * dx + dy * dy);
        
        // Reject if movement is too large (likely tracking error)
        if (movement > Config::getInstance().m_max_movement_distance) {

            std::cout << "[MOVEMENT OUTLIER] Feature ID " << feature_ids[i] << " movement: " << movement <<" from pixel: "<< prev_pts[i] << " to " << cur_pts[i] << std::endl;
            movement_outliers.push_back(feature_ids[i]);
            outlier_features.push_back(feature_ids[i]);
        }
    }

    // Step 2: Essential matrix RANSAC (more robust for VIO)
    std::vector<uchar> status;
    if (prev_pts.size() >= 5) {  // Essential matrix needs only 5 points minimum
        // Get camera intrinsic matrix from current frame
        double fx, fy, cx, cy;
        current_frame->get_camera_intrinsics(fx, fy, cx, cy);
        cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
        
        cv::Mat E = cv::findEssentialMat(prev_pts, cur_pts, camera_matrix, 
                                        cv::RANSAC, 0.95, 3.0, status);  // Use larger threshold for RANSAC
        
        // Check if Essential Matrix was successfully computed
        if (!E.empty() && E.rows == 3 && E.cols == 3) {
            // Calculate epipolar constraint errors manually
            std::vector<float> errors;
            
            // Calculate epipolar constraint errors manually
            for (size_t i = 0; i < prev_pts.size(); ++i) {
                // Convert to normalized coordinates
                cv::Point2f p1_norm = cv::Point2f((prev_pts[i].x - cx) / fx, (prev_pts[i].y - cy) / fy);
                cv::Point2f p2_norm = cv::Point2f((cur_pts[i].x - cx) / fx, (cur_pts[i].y - cy) / fy);
                
                // Calculate epipolar error: x2^T * E * x1
                cv::Mat x1 = (cv::Mat_<double>(3, 1) << p1_norm.x, p1_norm.y, 1.0);
                cv::Mat x2 = (cv::Mat_<double>(3, 1) << p2_norm.x, p2_norm.y, 1.0);
                cv::Mat error_mat = x2.t() * E * x1;
                float epipolar_error = std::abs(error_mat.at<double>(0, 0));
                errors.push_back(epipolar_error);
            }
            
            // Use error threshold instead of status (더 정밀한 제어)
            float error_threshold = 0.1f;  // Epipolar error threshold in normalized coordinates
            for (size_t i = 0; i < errors.size(); ++i) {
                if (errors[i] > error_threshold) {
                    ransac_outliers.push_back(feature_ids[i]);
                    // Check if not already marked as outlier
                    if (std::find(outlier_features.begin(), outlier_features.end(), feature_ids[i]) == outlier_features.end()) {
                        outlier_features.push_back(feature_ids[i]);
                    }
                }
            }
        } else {
            // Essential matrix computation failed, fall back to status-based rejection
            std::cout << "[ESSENTIAL] Failed to compute Essential Matrix, using status fallback" << std::endl;
            for (size_t i = 0; i < status.size(); ++i) {
                if (!status[i]) {
                    ransac_outliers.push_back(feature_ids[i]);
                    if (std::find(outlier_features.begin(), outlier_features.end(), feature_ids[i]) == outlier_features.end()) {
                        outlier_features.push_back(feature_ids[i]);
                    }
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
            
            if (vel_diff_x > Config::getInstance().m_max_velocity_change || 
                vel_diff_y > Config::getInstance().m_max_velocity_change) {
                velocity_outliers.push_back(feature_ids[i]);
                if (std::find(outlier_features.begin(), outlier_features.end(), feature_ids[i]) == outlier_features.end()) {
                    outlier_features.push_back(feature_ids[i]);
                }
            }
        }
    }

    // Remove all outliers
    if (!outlier_features.empty()) {
        spdlog::warn("----------------- DEBUG TRACKING OUTLIER -----------------");
        spdlog::warn("[OUTLIER] Movement outliers: {} features // threshold: {}", movement_outliers.size(), Config::getInstance().m_max_movement_distance);
        spdlog::warn("[OUTLIER] RANSAC outliers: {} features", ransac_outliers.size());
        spdlog::warn("[OUTLIER] Velocity outliers: {} features", velocity_outliers.size());
        spdlog::warn("[OUTLIER] Total removing: {} outlier features", outlier_features.size());
        spdlog::warn("----------------------------------------------------------");
    }
    for (int feature_id : outlier_features) {
        current_frame->remove_feature(feature_id);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
    
    // Log outlier rejection statistics (commented out for cleaner logs)
    // std::cout << "[OUTLIER REJECTION] " << prev_pts.size() << " features -> "
    //           << "Movement outliers: " << movement_outliers.size()
    //           << ", Essential outliers: " << ransac_outliers.size()
    //           << ", Velocity outliers: " << velocity_outliers.size()
    //           << ", Total removed: " << outlier_features.size()
    //           << ", Remaining: " << (prev_pts.size() - outlier_features.size()) << std::endl;

    // Debug output removed for cleaner logs
}

void FeatureTracker::set_mask(std::shared_ptr<Frame> frame) {
    if (frame->get_image().empty()) {
        std::cerr << "Cannot set mask: image is empty" << std::endl;
        return;
    }
    
    // Create mask initialized to 255 (valid areas for feature detection)
    m_mask = cv::Mat(frame->get_image().size(), CV_8UC1, cv::Scalar(255));
    
    const Config& config = Config::getInstance();
    int min_distance = static_cast<int>(config.m_min_distance);
    int border_size = config.m_border_size;
    
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
    
    // Debug output removed for cleaner logs
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

bool FeatureTracker::can_triangulate_feature(std::shared_ptr<Feature> feature, std::shared_ptr<Frame> frame) {
    if (!feature || !frame) {
        return false;
    }
    
    // Check if this is a stereo frame
    if (!frame->is_stereo()) {
        return false;
    }
    
    cv::Point2f pixel_coord = feature->get_pixel_coord();
    
    // Check if feature is within image bounds with some margin
    int border_margin = 10;
    if (pixel_coord.x < border_margin || pixel_coord.y < border_margin || 
        pixel_coord.x >= frame->get_image().cols - border_margin || 
        pixel_coord.y >= frame->get_image().rows - border_margin) {
        return false;
    }
    
    // Use Frame's stereo depth functionality to check if triangulation is feasible
    // This might be the slow part - let's time it
    auto stereo_check_start = std::chrono::high_resolution_clock::now();
    bool result = frame->has_valid_stereo_depth(pixel_coord);
    auto stereo_check_end = std::chrono::high_resolution_clock::now();
    auto stereo_check_time = std::chrono::duration_cast<std::chrono::microseconds>(stereo_check_end - stereo_check_start).count() / 1000.0;
    
    // Log if stereo depth check is very slow (raised threshold)
    if (stereo_check_time > 1.0) {
        std::cout << "[STEREO CHECK] Very slow stereo depth check: " << stereo_check_time << "ms" << std::endl;
    }
    
    return result;
}

std::shared_ptr<MapPoint> FeatureTracker::create_map_point_from_stereo(std::shared_ptr<Feature> feature, std::shared_ptr<Frame> frame) {
    if (!feature || !frame) {
        return nullptr;
    }
    
    // Check if feature has stereo match (already computed in batch)
    if (!feature->has_stereo_match()) {
        return nullptr;
    }
    
    cv::Point2f pixel_coord = feature->get_pixel_coord();
    cv::Point2f right_coord = feature->get_right_coord();
    
    // Get camera parameters
    double fx, fy, cx, cy;
    frame->get_camera_intrinsics(fx, fy, cx, cy);
    
    // Calculate depth from disparity
    float disparity = pixel_coord.x - right_coord.x;
    if (disparity <= 0) {
        return nullptr;
    }
    
    double depth = (fx * frame->get_baseline()) / disparity;
    
    // Check depth range
    if (depth <= 0 || depth < Config::getInstance().m_min_depth || 
        depth > Config::getInstance().m_max_depth) {
        return nullptr;
    }
    
    // Undistort the point
    cv::Point2f undistorted = frame->undistort_point(pixel_coord);
    
    // Unproject to 3D camera coordinates
    double x = (undistorted.x - cx) * depth / fx;
    double y = (undistorted.y - cy) * depth / fy;
    double z = depth;
    
    // Transform to world coordinates using frame pose
    Eigen::Matrix4f Twb = frame->get_Twb();
    Eigen::Vector4f camera_point(x, y, z, 1.0);
    Eigen::Vector4f world_point = Twb * camera_point;
    
    Eigen::Vector3f world_pos = world_point.head<3>();
    
    // Create new map point
    auto map_point = std::make_shared<MapPoint>(world_pos);
    
    return map_point;
}

int FeatureTracker::batch_stereo_matching_and_map_point_creation(const std::shared_ptr<Frame>& frame) {
    if (!frame->is_stereo()) {
        return 0;
    }

    // Collect all features without map points (both tracked and newly extracted)
    std::vector<cv::Point2f> left_points;
    std::vector<int> feature_indices;
    
    const auto& features = frame->get_features();
    for (int i = 0; i < features.size(); i++) {
        if (!frame->has_map_point(i)) {
            left_points.push_back(features[i]->get_pixel_coord());
            feature_indices.push_back(i);
        }
    }
    
    if (left_points.empty()) {
        return 0;
    }
    
    std::cout << "[BATCH STEREO] Processing " << left_points.size() << " features without map points" << std::endl;
    
    // Use optical flow to find corresponding points in right image
    std::vector<cv::Point2f> right_points;
    std::vector<uchar> status;
    std::vector<float> error;
    
    cv::calcOpticalFlowPyrLK(
        frame->get_left_image(), 
        frame->get_right_image(),
        left_points, 
        right_points,
        status, 
        error,
        cv::Size(21, 21), 
        3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        0, 
        0.001
    );
    
    int successful_matches = 0;
    int new_map_points_created = 0;
    
    for (int i = 0; i < left_points.size(); i++) {
        if (status[i]) {  // Error threshold
            double disparity = left_points[i].x - right_points[i].x;
            
            // Check disparity constraints
            if (disparity > 0) {
                
                successful_matches++;
                
                // Create map point for this feature
                int feature_idx = feature_indices[i];
                auto feature = features[feature_idx];
                
                // Set stereo match info on the feature
                feature->set_stereo_match(right_points[i], disparity);
                
                // Create map point from stereo
                auto map_point = create_map_point_from_stereo(feature, frame);
                if (map_point && !map_point->is_bad()) {
                    // Associate map point with feature
                    frame->set_map_point(feature_idx, map_point);
                    map_point->add_observation(frame, feature_idx);
                    new_map_points_created++;
                }
            }
        }
    }
    
    std::cout << "[BATCH STEREO] Successfully matched " << successful_matches 
              << "/" << left_points.size() << " features, created " 
              << new_map_points_created << " map points" << std::endl;
    
    return new_map_points_created;
}

//...existing code...
} // namespace lightweight_vio
