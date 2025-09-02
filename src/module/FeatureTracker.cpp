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

    // Note: Stereo matching is now handled by Frame::compute_stereo_depth() in Estimator
    // This avoids duplicate stereo matching operations
    
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
    
    // After tracking and stereo matching, check if we need more features
    int current_features = current_frame->get_feature_count();
    if (current_features < m_config.m_max_features) {
        // Update mask to avoid detecting features near existing ones
        set_mask(current_frame);
        
        // Extract new features to reach max_features
        auto [new_features, new_successful_matches] = extract_new_features(current_frame);
        new_extracted_features += new_features;
        new_map_points_from_extraction += new_successful_matches;
    }
    
    // Grid-based feature selection during tracking + additional feature extraction
    // ensures total feature count reaches max_features while maintaining good distribution
    
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
        
        // Create all features without map points - Estimator will handle map point creation during keyframe creation
        int next_feature_id = frame->get_feature_count();  // Start from current count
        for (const auto& corner : corners) {
            auto feature = std::make_shared<Feature>(next_feature_id++, corner);
            frame->add_feature(feature);
        }
        
        // Return feature count and 0 for map points (no immediate map point creation)
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

    // Extract points from previous frame features using grid-based selection
    auto point_extraction_start = std::chrono::high_resolution_clock::now();
    
    // Get selected feature indices from grid-based selection
    std::vector<int> feature_indices = select_features_for_tracking(previous_frame);
    
    std::vector<cv::Point2f> prev_pts;
    std::vector<int> valid_feature_indices;  // Track valid feature indices for tracking
    
    const auto& prev_features = previous_frame->get_features();
    for (int idx : feature_indices) {
        if (idx >= 0 && idx < prev_features.size() && prev_features[idx] && prev_features[idx]->is_valid()) {
            prev_pts.push_back(prev_features[idx]->get_pixel_coord());
            valid_feature_indices.push_back(idx);  // Store the original index
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
                            Config::getInstance().term_criteria(),
                            0, 0.001);    auto flow_end = std::chrono::high_resolution_clock::now();
    auto flow_time = std::chrono::duration_cast<std::chrono::microseconds>(flow_end - flow_start).count() / 1000.0;

    // Create features for current frame based on tracking results
    auto feature_creation_start = std::chrono::high_resolution_clock::now();
    int tracked_features = 0;
    int associated_map_points = 0;
    int new_map_points_created = 0;
    int next_feature_id = current_frame->get_feature_count();  // Start from current count
    
    // Tracking failure counters for debugging
    int optical_flow_failed = 0;
    int border_failed = 0;
    int error_threshold_failed = 0;
    int total_attempted = prev_pts.size();
    
    for (size_t i = 0; i < prev_pts.size(); ++i) {
        if (!status[i]) {
            optical_flow_failed++;
            continue;
        }
        
        if (!is_in_border(cur_pts[i], current_frame->get_image().size())) {
            border_failed++;
            continue;
        }
        
        // Apply optical flow quality checks using config parameters
        float dx = cur_pts[i].x - prev_pts[i].x;
        float dy = cur_pts[i].y - prev_pts[i].y;
        float movement = std::sqrt(dx*dx + dy*dy);
        
        // Check error threshold
        if (err[i] >= Config::getInstance().m_error_threshold) {
            error_threshold_failed++;
            continue;
        }
        
        // Get the original feature index
        int original_feature_idx = valid_feature_indices[i];
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
            
            // Map point creation is now handled only by Estimator during keyframe creation
        } 
    }
    auto feature_creation_end = std::chrono::high_resolution_clock::now();
    auto feature_creation_time = std::chrono::duration_cast<std::chrono::microseconds>(feature_creation_end - feature_creation_start).count() / 1000.0;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
    
    // Debug output for tracking failures
    if (m_config.m_enable_debug_output || total_attempted > 0) {
        int total_failed = optical_flow_failed + border_failed + error_threshold_failed;
        float success_rate = total_attempted > 0 ? (float)tracked_features / total_attempted * 100.0f : 0.0f;
        
        spdlog::info("[TRACKING DEBUG] Frame-to-frame tracking results:");
        spdlog::info("  Attempted: {} features", total_attempted);
        spdlog::info("  Successful: {} features ({:.1f}%)", tracked_features, success_rate);
        spdlog::info("  Failed breakdown:");
        spdlog::info("    - Optical flow failed: {} ({:.1f}%)", optical_flow_failed, 
                     total_attempted > 0 ? (float)optical_flow_failed / total_attempted * 100.0f : 0.0f);
        spdlog::info("    - Out of border: {} ({:.1f}%)", border_failed,
                     total_attempted > 0 ? (float)border_failed / total_attempted * 100.0f : 0.0f);
        spdlog::info("    - Error threshold (>{:.1f}): {} ({:.1f}%)", 
                     Config::getInstance().m_error_threshold, error_threshold_failed,
                     total_attempted > 0 ? (float)error_threshold_failed / total_attempted * 100.0f : 0.0f);
        spdlog::info("  Map point associations: {}", associated_map_points);
    }
    
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

void FeatureTracker::manage_grid_based_features(std::shared_ptr<Frame> frame) {
    // Skip grid management for very first frame or frames with no features
    if (!frame || frame->get_feature_count() == 0) {
        return;
    }
    
    // Initialize temporary 2D grid to store feature indices
    const int grid_rows = m_config.m_grid_rows;
    const int grid_cols = m_config.m_grid_cols;
    
    std::vector<std::vector<std::vector<int>>> temp_grid(grid_rows, 
                                                        std::vector<std::vector<int>>(grid_cols));
    
    // Assign features to grid cells
    assign_features_to_grid(frame, temp_grid);
    
    // Limit features per grid based on max_features_per_grid
    limit_features_per_grid(frame, temp_grid);
}

void FeatureTracker::assign_features_to_grid(std::shared_ptr<Frame> frame, 
                                             std::vector<std::vector<std::vector<int>>>& temp_grid) {
    const int grid_cols = m_config.m_grid_cols;
    const int grid_rows = m_config.m_grid_rows;
    const int img_width = m_config.m_image_width;
    const int img_height = m_config.m_image_height;
    
    // Assign each feature to its corresponding grid cell
    const auto& features = frame->get_features();
    for (size_t i = 0; i < features.size(); ++i) {
        const auto& feature = features[i];
        if (!feature) continue;
        
        // Calculate grid coordinates
        float cell_width = (float)img_width / grid_cols;
        float cell_height = (float)img_height / grid_rows;
        
        cv::Point2f pixel_coord = feature->get_pixel_coord();
        
        // Safety check for pixel coordinates
        if (pixel_coord.x < 0 || pixel_coord.x >= img_width || 
            pixel_coord.y < 0 || pixel_coord.y >= img_height) {
            continue; // Skip features outside image bounds
        }
        
        int grid_x = std::min((int)(pixel_coord.x / cell_width), grid_cols - 1);
        int grid_y = std::min((int)(pixel_coord.y / cell_height), grid_rows - 1);
        
        // Additional safety check for grid indices
        if (grid_x < 0 || grid_x >= grid_cols || grid_y < 0 || grid_y >= grid_rows) {
            continue; // Skip invalid grid coordinates
        }
        
        // Add feature index to the corresponding grid cell
        temp_grid[grid_y][grid_x].push_back(i);
    }
}

void FeatureTracker::assign_features_to_grid_with_indices(std::shared_ptr<Frame> frame, 
                                                          std::vector<std::vector<std::vector<int>>>& temp_grid,
                                                          const std::vector<int>& valid_indices) {
    const int grid_cols = m_config.m_grid_cols;
    const int grid_rows = m_config.m_grid_rows;
    const int img_width = m_config.m_image_width;
    const int img_height = m_config.m_image_height;
    
    // Assign only specified features to their corresponding grid cells
    const auto& features = frame->get_features();
    for (int feature_idx : valid_indices) {
        if (feature_idx < 0 || feature_idx >= features.size()) {
            continue; // Skip invalid indices
        }
        
        const auto& feature = features[feature_idx];
        if (!feature) continue;
        
        // Calculate grid coordinates
        float cell_width = (float)img_width / grid_cols;
        float cell_height = (float)img_height / grid_rows;
        
        cv::Point2f pixel_coord = feature->get_pixel_coord();
        
        // Safety check for pixel coordinates
        if (pixel_coord.x < 0 || pixel_coord.x >= img_width || 
            pixel_coord.y < 0 || pixel_coord.y >= img_height) {
            continue; // Skip features outside image bounds
        }
        
        int grid_x = std::min((int)(pixel_coord.x / cell_width), grid_cols - 1);
        int grid_y = std::min((int)(pixel_coord.y / cell_height), grid_rows - 1);
        
        // Additional safety check for grid indices
        if (grid_x < 0 || grid_x >= grid_cols || grid_y < 0 || grid_y >= grid_rows) {
            continue; // Skip invalid grid coordinates
        }
        
        // Add feature index to the corresponding grid cell
        temp_grid[grid_y][grid_x].push_back(feature_idx);
    }
}

void FeatureTracker::limit_features_per_grid(std::shared_ptr<Frame> frame, 
                                             std::vector<std::vector<std::vector<int>>>& temp_grid) {
    const int max_features_per_grid = m_config.m_max_features_per_grid;
    auto& features = frame->get_features_mutable();
    auto& map_points = frame->get_map_points_mutable();  // Use mutable version
    
    // Process each grid cell
    for (size_t row = 0; row < temp_grid.size(); ++row) {
        for (size_t col = 0; col < temp_grid[row].size(); ++col) {
            auto& cell_features = temp_grid[row][col];
            
            // Skip if this cell has fewer features than the limit
            if (cell_features.size() <= max_features_per_grid) {
                continue;
            }
            
            // Sort features by track count (as a proxy for strength) in descending order
            // Note: Using track_count since corner_response is not available
            std::sort(cell_features.begin(), cell_features.end(), 
                     [&features](int a, int b) {
                         // Safety checks for valid indices and non-null features
                         if (a >= features.size() || b >= features.size()) return false;
                         if (!features[a] || !features[b]) return false;
                         return features[a]->get_track_count() > features[b]->get_track_count();
                     });
            
            // Keep only the strongest max_features_per_grid features
            // Disconnect and remove the rest from their map points
            std::vector<int> features_to_remove;
            for (size_t i = max_features_per_grid; i < cell_features.size(); ++i) {
                int feature_idx = cell_features[i];
                
                // Safety check for valid feature index
                if (feature_idx < 0 || feature_idx >= features.size()) {
                    continue; // Skip invalid indices
                }
                
                features_to_remove.push_back(feature_idx);
                
                // Disconnect from map point by removing observation and setting to nullptr
                if (feature_idx < map_points.size()) {
                    auto mp = map_points[feature_idx];
                    if (mp) {
                        // Remove this frame's observation from the map point
                        mp->remove_observation(frame);
                        // Set map point to nullptr in frame
                        map_points[feature_idx] = nullptr;
                    }
                }
            }
            
            // Mark features for removal by setting them to nullptr
            for (int idx : features_to_remove) {
                if (idx >= 0 && idx < features.size()) {
                    features[idx] = nullptr;
                }
            }
            
            // Keep only the selected features in the cell
            cell_features.resize(max_features_per_grid);
        }
    }
    
    if (m_config.m_enable_debug_output) {
        // Count remaining valid and connected features
        auto& features = frame->get_features_mutable();
        auto& map_points = frame->get_map_points_mutable();
        
        int valid_features = 0;
        int connected_features = 0;
        
        size_t max_size = std::min(features.size(), map_points.size());
        
        for (size_t i = 0; i < features.size(); ++i) {
            if (features[i] != nullptr) {
                valid_features++;
                if (i < map_points.size() && map_points[i] && !map_points[i]->is_bad()) {
                    connected_features++;
                }
            }
        }
        
        spdlog::info("Grid-based feature management: {} valid features, {} connected to map points", 
                     valid_features, connected_features);
    }
}

std::vector<int> FeatureTracker::select_features_for_tracking(std::shared_ptr<Frame> previous_frame) {
    std::vector<int> selected_indices;
    
    if (!previous_frame || previous_frame->get_feature_count() == 0) {
        return selected_indices;
    }
    
    // First, filter out features that have high observation count but no map point
    const int max_observation_threshold = m_config.m_max_observation_without_mappoint;
    const auto& features = previous_frame->get_features();
    std::vector<int> valid_feature_indices;
    int filtered_count = 0;
    
    for (int i = 0; i < features.size(); ++i) {
        if (!features[i] || !features[i]->is_valid()) {
            continue;
        }
        
        // Check if feature has high observation count but no associated map point
        int track_count = features[i]->get_track_count();
        bool has_map_point = previous_frame->has_map_point(i) && 
                            previous_frame->get_map_point(i) && 
                            !previous_frame->get_map_point(i)->is_bad();
        
        if (track_count >= max_observation_threshold && !has_map_point) {
            // Filter out this feature - it's been observed many times but never became a map point
            filtered_count++;
            if (m_config.m_enable_debug_output && filtered_count <= 5) {
                spdlog::debug("Filtering feature {} with {} observations but no map point", 
                             features[i]->get_feature_id(), track_count);
            }
            continue;
        }
        
        valid_feature_indices.push_back(i);
    }
    
    if (m_config.m_enable_debug_output && filtered_count > 0) {
        spdlog::info("Filtered {} features with ≥{} observations but no map point", 
                     filtered_count, max_observation_threshold);
    }
    
    // Initialize grid
    const int grid_rows = m_config.m_grid_rows;
    const int grid_cols = m_config.m_grid_cols;
    const int max_features_per_grid = m_config.m_max_features_per_grid;
    
    std::vector<std::vector<std::vector<int>>> temp_grid(grid_rows, 
                                                        std::vector<std::vector<int>>(grid_cols));
    
    // Assign only valid features to grid cells
    assign_features_to_grid_with_indices(previous_frame, temp_grid, valid_feature_indices);
    
    // Select best features from each grid cell
    for (size_t row = 0; row < temp_grid.size(); ++row) {
        for (size_t col = 0; col < temp_grid[row].size(); ++col) {
            auto& cell_features = temp_grid[row][col];
            
            if (cell_features.empty()) {
                continue;
            }
            
            const auto& features = previous_frame->get_features();
            
            // Sort features by track count (as a proxy for strength) in descending order
            std::sort(cell_features.begin(), cell_features.end(), 
                     [&features](int a, int b) {
                         // Safety checks for valid indices and non-null features
                         if (a >= features.size() || b >= features.size()) return false;
                         if (!features[a] || !features[b]) return false;
                         return features[a]->get_track_count() > features[b]->get_track_count();
                     });
            
            // Select up to max_features_per_grid best features from this cell
            int features_to_select = std::min((int)cell_features.size(), max_features_per_grid);
            for (int i = 0; i < features_to_select; ++i) {
                int feature_idx = cell_features[i];
                if (feature_idx >= 0 && feature_idx < features.size() && features[feature_idx]) {
                    selected_indices.push_back(feature_idx);
                }
            }
        }
    }
    
    if (m_config.m_enable_debug_output) {
        spdlog::info("Grid-based feature selection: {} features selected for tracking from {} valid (filtered {} high-observation features without map points)", 
                     selected_indices.size(), valid_feature_indices.size(), filtered_count);
        // spdlog::info("Max features per grid: {}, Grid size: {}x{} = {} cells", 
        //              max_features_per_grid, grid_cols, grid_rows, grid_cols * grid_rows);
        // spdlog::info("Theoretical max features: {} ({}x{}x{})", 
        //              grid_cols * grid_rows * max_features_per_grid, grid_cols, grid_rows, max_features_per_grid);
    }
    
    return selected_indices;
}

} // namespace lightweight_vio
