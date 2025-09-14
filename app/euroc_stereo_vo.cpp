/**
 * @file      euroc_stereo_vo.cpp
 * @brief     Main application entry point for the EuRoC stereo VO pipeline.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-08-11
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <sstream>
#include <memory>
#include <chrono>
#include <thread>
#include <cstdio>
#include <numeric>

#include <database/Frame.h>
#include <database/Feature.h>
#include <database/MapPoint.h>
#include <processing/FeatureTracker.h>
#include <processing/Estimator.h>
#include <util/Config.h>
#include <util/EurocUtils.h>
#include <viewer/PangolinViewer.h>

using namespace lightweight_vio;

struct ImageData {
    long long timestamp;
    std::string filename;
};

// Helper function to trim whitespace
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

// Helper function to extract positions from poses for legacy trajectory update
std::vector<Eigen::Vector3f> extract_positions_from_poses(const std::vector<Eigen::Matrix4f>& poses) {
    std::vector<Eigen::Vector3f> positions;
    for (const auto& pose : poses) {
        positions.push_back(pose.block<3, 1>(0, 3));
    }
    return positions;
}

std::vector<ImageData> load_image_timestamps(const std::string& dataset_path) {
    std::vector<ImageData> image_data;
    std::string data_file = dataset_path + "/mav0/cam0/data.csv";
    
    std::ifstream file(data_file);
    if (!file.is_open()) {
        spdlog::error("[Dataset] Cannot open data.csv file: {}", data_file);
        return image_data;
    }
    
    std::string line;
    std::getline(file, line); // Skip header
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::stringstream ss(line);
        std::string timestamp_str, filename;
        
        if (std::getline(ss, timestamp_str, ',') && std::getline(ss, filename)) {
            ImageData data;
            data.timestamp = std::stoll(trim(timestamp_str));
            data.filename = trim(filename);
            image_data.push_back(data);
        }
    }
    
    // Removed dataset loading log
    return image_data;
}

cv::Mat load_image(const std::string& dataset_path, const std::string& filename, int cam_id = 0) {
    std::string cam_folder = (cam_id == 0) ? "cam0" : "cam1";
    std::string full_path = dataset_path + "/mav0/" + cam_folder + "/data/" + filename;
    cv::Mat image = cv::imread(full_path, cv::IMREAD_GRAYSCALE);
    
    if (image.empty()) {
        spdlog::error("[Dataset] Cannot load image: {}", full_path);
    }
    
    return image;
}

std::pair<cv::Mat, cv::Mat> load_stereo_images(const std::string& dataset_path, const std::string& filename) {
    cv::Mat left_image = load_image(dataset_path, filename, 0);
    cv::Mat right_image = load_image(dataset_path, filename, 1);
    return std::make_pair(left_image, right_image);
}

std::vector<Eigen::Vector3f> extract_3d_points(std::shared_ptr<Frame> frame) {
    std::vector<Eigen::Vector3f> points;
    
    if (!frame) return points;
    
    for (const auto& feature : frame->get_features()) {
        if (feature->has_3d_point()) {
            points.push_back(feature->get_3d_point());
        }
    }
    
    return points;
}

std::vector<Eigen::Vector3f> extract_all_map_points(const Estimator& estimator) {
    std::vector<Eigen::Vector3f> points;
    
    const auto& map_points = estimator.get_map_points();
    
    for (const auto& mp : map_points) {
        if (mp && !mp->is_bad()) {
            Eigen::Vector3f position = mp->get_position();
            points.push_back(position);
        }
    }
    
    return points;
}

std::vector<Eigen::Vector3f> extract_current_frame_map_points(const Estimator& estimator) {
    std::vector<Eigen::Vector3f> points;
    
    auto current_frame = estimator.get_current_frame();
    if (!current_frame) {
        return points;
    }
    
    const auto& frame_map_points = current_frame->get_map_points();
    
    for (const auto& mp : frame_map_points) {
        if (mp && !mp->is_bad()) {
            Eigen::Vector3f position = mp->get_position();
            points.push_back(position);
        }
    }
    
    return points;
}

std::vector<Eigen::Vector3f> extract_all_accumulated_map_points(const Estimator& estimator) {
    std::vector<Eigen::Vector3f> points;
    
    const auto& all_map_points = estimator.get_map_points();
    
    int valid_count = 0;
    for (const auto& mp : all_map_points) {
        if (mp && !mp->is_bad()) {
            Eigen::Vector3f position = mp->get_position();
            points.push_back(position);
            valid_count++;
        }
    }
    
    return points;
}

int main(int argc, char* argv[]) {
    // Initialize spdlog for immediate colored output
    spdlog::set_level(spdlog::level::debug);  // Enable debug messages for timestamp matching
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    
    if (argc != 3) {
        spdlog::error("Usage: {} <config_file_path> <euroc_dataset_path>", argv[0]);
        spdlog::error("Example: {} config/euroc.yaml /path/to/MH_01_easy", argv[0]);
        return -1;
    }
    
    std::string config_path = argv[1];
    std::string dataset_path = argv[2];
    
    // Load configuration
    try {
        Config::getInstance().load(config_path);
        spdlog::info("[Config] Successfully loaded configuration from: {}", config_path);
    } catch (const std::exception& e) {
        spdlog::error("[Config] Failed to load configuration from {}: {}", config_path, e.what());
        return -1;
    }
    
    // Load EuRoC ground truth data
    if (!lightweight_vio::EurocUtils::load_ground_truth(dataset_path)) {
        spdlog::warn("[EuRoC] Failed to load ground truth data, continuing without it");
    }
    
    // Load image timestamps
    std::vector<ImageData> image_data = load_image_timestamps(dataset_path);
    if (image_data.empty()) {
        spdlog::error("[Dataset] No images found in dataset");
        return -1;
    }
    
    // Pre-match image timestamps with ground truth and get valid range
    size_t start_frame_idx = 0;
    size_t end_frame_idx = image_data.size();
    
    if (lightweight_vio::EurocUtils::has_ground_truth()) {
        std::vector<long long> image_timestamps;
        image_timestamps.reserve(image_data.size());
        for (const auto& img : image_data) {
            image_timestamps.push_back(img.timestamp);
        }
        
        if (lightweight_vio::EurocUtils::match_image_timestamps(image_timestamps)) {
            size_t matched_count = lightweight_vio::EurocUtils::get_matched_count();
            
            // Find the valid frame range based on matched timestamps
            if (matched_count > 0) {
                // Get the first and last matched image timestamps
                long long first_matched_ts = lightweight_vio::EurocUtils::get_matched_timestamp(0);
                long long last_matched_ts = lightweight_vio::EurocUtils::get_matched_timestamp(matched_count - 1);
                
                // Find corresponding indices in original image_data
                for (size_t i = 0; i < image_data.size(); ++i) {
                    if (image_data[i].timestamp == first_matched_ts) {
                        start_frame_idx = i;
                        break;
                    }
                }
                
                for (size_t i = image_data.size(); i > 0; --i) {
                    if (image_data[i-1].timestamp == last_matched_ts) {
                        end_frame_idx = i;
                        break;
                    }
                }
            }
        }
    }
    
    // Initialize 3D viewer (optional)
    PangolinViewer* viewer = nullptr;
    std::unique_ptr<PangolinViewer> viewer_ptr = std::make_unique<PangolinViewer>();
    if (viewer_ptr->initialize(1920*2, 1080*2)) {  // Pangolin 뷰어 초기화
        viewer = viewer_ptr.get();
        spdlog::info("[Viewer] Pangolin viewer initialized successfully");
        
        // Wait for viewer to be fully ready
        spdlog::info("[Viewer] Waiting for viewer to be fully ready...");
        while (viewer && !viewer->is_ready()) {
            viewer->render();
            std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
        }
        spdlog::info("[Viewer] Viewer is ready!");
    } else {
        spdlog::warn("[Viewer] Failed to initialize 3D viewer, running without visualization");
    }
    
    // Initialize Estimator
    Estimator estimator;
    
    // Set initial ground truth pose for first frame if available
    if (lightweight_vio::EurocUtils::has_ground_truth() && !image_data.empty()) {
        auto first_gt_pose = lightweight_vio::EurocUtils::get_matched_pose(0);
        if (first_gt_pose.has_value()) {
            estimator.set_initial_gt_pose(first_gt_pose.value());
            spdlog::info("[GT_INIT] Set initial ground truth pose for VIO estimation");
        }
    }
    
    // Control variables for step mode
    bool auto_play = true;
    bool step_mode = false;
    bool advance_frame = false;
    
    // Vector to store ground truth poses for trajectory export
    std::vector<Eigen::Matrix4f> gt_poses;
    
    // Vector to store frame processing times for statistics
    std::vector<double> frame_processing_times;

    // Variables to store statistics for combined output
    bool transform_stats_available = false;
    double transform_rot_mean, transform_rot_median, transform_rot_min, transform_rot_max, transform_rot_rmse;
    double transform_trans_mean, transform_trans_median, transform_trans_min, transform_trans_max, transform_trans_rmse;
    double transform_lin_vel_mean, transform_lin_vel_median, transform_lin_vel_min, transform_lin_vel_max;
    double transform_ang_vel_mean, transform_ang_vel_median, transform_ang_vel_min, transform_ang_vel_max;
    size_t transform_total_pairs, transform_total_frames, transform_total_gt_poses;
    bool velocity_stats_available = false;

    spdlog::info("[VO] frame {} to frame {} (GT-matched range)", start_frame_idx, end_frame_idx);
    
    // Process frames in the valid GT range
    size_t current_idx = start_frame_idx;
    size_t processed_frames = 0;
    while (current_idx < end_frame_idx) {
        // Check if viewer wants to exit
        if (viewer && viewer->should_close()) {
            spdlog::info("[Viewer] User requested exit");
            break;
        }
        
        // Check if finish button was pressed
        if (viewer && viewer->is_finish_requested()) {
            spdlog::info("[Viewer] User pressed Finish & Exit button");
            break;
        }
        
        // Process keyboard input if viewer is available
        if (viewer) {
            viewer->process_keyboard_input(auto_play, step_mode, advance_frame);
            viewer->sync_ui_state(auto_play, step_mode);  // Sync UI checkbox state
        }
        
        // In step mode, wait for user input before processing
        if (step_mode && !advance_frame) {
            if (viewer) {
                viewer->render(); // Keep rendering while waiting
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            continue; // Wait for user input
        }
        
        auto frame_start = std::chrono::high_resolution_clock::now();
        
        // Load stereo images
        cv::Mat left_image = load_image(dataset_path, image_data[current_idx].filename, 0);
        cv::Mat right_image = load_image(dataset_path, image_data[current_idx].filename, 1);
        
        if (left_image.empty()) {
            spdlog::warn("[Dataset] Skipping frame {} due to empty image", current_idx);
            ++current_idx;  // Move to next frame even if current is empty
            continue;
        }
        
        auto preprocess_start = std::chrono::high_resolution_clock::now();
        
        // Image preprocessing with enhanced illumination handling
        cv::Mat processed_left_image, processed_right_image;
        
        // Step 1: Global Histogram Equalization for overall brightness normalization
        cv::Mat equalized_left, equalized_right;
        cv::equalizeHist(left_image, equalized_left);
        
        // Step 2: CLAHE for local contrast enhancement
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(equalized_left, processed_left_image);
        
        if (!right_image.empty()) {
            cv::equalizeHist(right_image, equalized_right);
            clahe->apply(equalized_right, processed_right_image);
        }
        
        auto estimation_start = std::chrono::high_resolution_clock::now();
        
        // Process frame through estimator
        Estimator::EstimationResult result = estimator.process_frame(
            processed_left_image, processed_right_image, 
            image_data[current_idx].timestamp);
        


     
            
        // Handle ground truth pose for comparison
        // processed_frames는 GT 매칭에서의 인덱스와 직접 대응됨
        if (lightweight_vio::EurocUtils::get_matched_count() > processed_frames) {
            auto gt_pose_opt = lightweight_vio::EurocUtils::get_matched_pose(processed_frames);
            if (gt_pose_opt.has_value()) {
                Eigen::Matrix4f gt_pose = gt_pose_opt.value();
                
                // Store ground truth pose for trajectory export
                gt_poses.push_back(gt_pose);
                
                // Debug first few GT poses
                if (processed_frames < 3) {
                    long long matched_ts = lightweight_vio::EurocUtils::get_matched_timestamp(processed_frames);
                    spdlog::info("[GT_DEBUG] processed_frames={}, matched_ts={}, gt_pos=({:.6f},{:.6f},{:.6f})", 
                                processed_frames, matched_ts, gt_pose(0,3), gt_pose(1,3), gt_pose(2,3));
                }
                
                // Always add GT pose to viewer trajectory for comparison
                if (viewer) {
                    viewer->add_ground_truth_pose(gt_pose);
                }
                
                // Log comparison between VIO estimation and GT (reduced frequency)
                auto current_frame = estimator.get_current_frame();
                if (current_frame && processed_frames % 10 == 0) {
                    Eigen::Matrix4f vio_estimated_pose = current_frame->get_Twb();
                    
                    // Extract positions for comparison
                    Eigen::Vector3f gt_position = gt_pose.block<3,1>(0,3);
                    Eigen::Vector3f vio_position = vio_estimated_pose.block<3,1>(0,3);
                    
                    float position_error = (gt_position - vio_position).norm();
                    spdlog::info("[GT_COMP] Frame {}: VIO_pos=({:.3f},{:.3f},{:.3f}), GT_pos=({:.3f},{:.3f},{:.3f}), ERROR={:.3f}m", 
                                processed_frames, 
                                vio_position.x(), vio_position.y(), vio_position.z(),
                                gt_position.x(), gt_position.y(), gt_position.z(),
                                position_error);
                }
            }
        }
        
            
        auto frame_end = std::chrono::high_resolution_clock::now();
        
        // Calculate frame processing time
        auto frame_duration = std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start);
        double frame_time_ms = frame_duration.count() / 1000.0; // Convert to milliseconds
        frame_processing_times.push_back(frame_time_ms);
        
        // Update viewer if available
        if (viewer) {
            auto current_frame = estimator.get_current_frame();
            if (current_frame) {
                // Update current pose
                Eigen::Matrix4f current_pose = current_frame->get_Twb();
                viewer->update_pose(current_pose);
                
                // Update current frame camera pose (T_wc) - same as keyframes
                Eigen::Matrix4f current_camera_pose = current_frame->get_Twc();
                viewer->update_camera_pose(current_camera_pose);
                
                // Update trajectory with both estimated and ground truth
                static std::vector<Eigen::Matrix4f> trajectory_poses;  // Changed to Matrix4f for debug
                static std::vector<Eigen::Matrix4f> gt_trajectory_poses;  // Changed to Matrix4f
                trajectory_poses.push_back(current_pose);  // Store full pose instead of just position
                
                // Add ground truth point if available
                if (lightweight_vio::EurocUtils::get_matched_count() > processed_frames) {
                    auto gt_pose_opt = lightweight_vio::EurocUtils::get_matched_pose(processed_frames);
                    if (gt_pose_opt.has_value()) {
                        Eigen::Matrix4f gt_pose = gt_pose_opt.value();
                        gt_trajectory_poses.push_back(gt_pose);  // Store full pose matrix
                        
                        // Update viewer with both trajectories (prints comparison inside)
                        viewer->update_trajectory_with_gt(trajectory_poses, gt_trajectory_poses);
                    } else {
                        // Only estimated trajectory available
                        viewer->update_trajectory(extract_positions_from_poses(trajectory_poses));
                    }
                } else {
                    // Only estimated trajectory available
                    viewer->update_trajectory(extract_positions_from_poses(trajectory_poses));
                }
                
                // Add current frame to viewer for sliding window management
                viewer->add_frame(current_frame);
                
                // Update keyframe window with latest keyframes from estimator
                const auto keyframes = estimator.get_keyframes_safe();  // Use thread-safe version
                viewer->update_keyframe_window(keyframes);
                
                // Update last keyframe if we have keyframes
                if (!keyframes.empty()) {
                    viewer->set_last_keyframe(keyframes.back());
                    
                    // Calculate relative pose from last keyframe to current frame
                    Eigen::Matrix4f last_keyframe_pose = keyframes.back()->get_Twb();
                    Eigen::Matrix4f current_pose_matrix = current_frame->get_Twb();
                    Eigen::Matrix4f relative_pose = last_keyframe_pose.inverse() * current_pose_matrix;
                    viewer->update_relative_pose_from_last_keyframe(relative_pose);
                }
                
                // Update map points with sliding window approach
                // Get all map points from estimator
                std::vector<std::shared_ptr<MapPoint>> all_map_points_shared;
                std::vector<std::shared_ptr<MapPoint>> window_map_points_shared;
                
                // Collect all unique map points from all frames
                std::set<std::shared_ptr<MapPoint>> unique_map_points;
                for (const auto& kf : keyframes) {
                    const auto& kf_map_points = kf->get_map_points();
                    for (const auto& mp : kf_map_points) {
                        if (mp && !mp->is_bad()) {
                            unique_map_points.insert(mp);
                        }
                    }
                }
                
                // Convert to vectors
                all_map_points_shared.assign(unique_map_points.begin(), unique_map_points.end());
                
                // Window map points are from current sliding window keyframes
                for (const auto& kf : keyframes) {
                    const auto& kf_map_points = kf->get_map_points();
                    for (const auto& mp : kf_map_points) {
                        if (mp && !mp->is_bad()) {
                            window_map_points_shared.push_back(mp);
                        }
                    }
                }
                
                viewer->update_all_map_points(all_map_points_shared);
                viewer->update_window_map_points(window_map_points_shared);
                
                // For legacy support, update current frame tracking points
                std::vector<Eigen::Vector3f> current_map_points = extract_current_frame_map_points(estimator);
                viewer->update_map_points({}, current_map_points);  // Empty all_points, only current points
                
                // Update tracking image
                static std::shared_ptr<Frame> previous_frame = nullptr;
                cv::Mat tracking_image;
                if (previous_frame) {
                    tracking_image = current_frame->draw_tracks(*previous_frame);
                } else {
                    tracking_image = current_frame->draw_features();
                }
                
                // Calculate frame statistics
                int tracked_features = 0;
                int new_features = 0;
                int stereo_matches = 0;
                int map_points_count = 0;
                
                for (const auto& feature : current_frame->get_features()) {
                    if (feature->has_tracked_feature()) {
                        tracked_features++;
                    } else {
                        new_features++;
                    }
                }
                
                // Count stereo matches and map points
                const auto& map_points = current_frame->get_map_points();
                for (const auto& mp : map_points) {
                    if (mp && !mp->is_bad()) {
                        map_points_count++;
                    }
                }
                
                // Estimate stereo matches from available data
                stereo_matches = map_points_count; // Approximation - actual stereo matches might be slightly different
                
                // Calculate success rate
                int total_features = current_frame->get_feature_count();
                float success_rate = (total_features > 0) ? 
                    (static_cast<float>(stereo_matches) / static_cast<float>(total_features)) * 100.0f : 0.0f;
                
                // Get position error if ground truth is available
                float position_error = 0.0f;
                if (lightweight_vio::EurocUtils::get_matched_count() > processed_frames) {
                    auto gt_pose_opt = lightweight_vio::EurocUtils::get_matched_pose(processed_frames);
                    if (gt_pose_opt.has_value()) {
                        Eigen::Matrix4f gt_pose = gt_pose_opt.value();
                        Eigen::Matrix4f estimated_pose = current_frame->get_Twb();
                        
                        Eigen::Vector3f gt_position = gt_pose.block<3,1>(0,3);
                        Eigen::Vector3f est_position = estimated_pose.block<3,1>(0,3);
                        
                        position_error = (gt_position - est_position).norm();
                    }
                }
                
                // Update tracking statistics in viewer
                viewer->update_tracking_stats(
                    current_idx + 1,           // frame_id
                    total_features,            // total_features  
                    stereo_matches,            // stereo_matches
                    map_points_count,          // map_points
                    success_rate,              // success_rate
                    position_error             // position_error
                );
                
                // Update viewer frame information (legacy)
                viewer->update_frame_info(current_idx + 1, total_features, tracked_features, new_features);
                
                // Update tracking image with map point indices
                if (current_frame) {
                    const auto& features = current_frame->get_features();
                    const auto& map_points = current_frame->get_map_points();
                    viewer->update_tracking_image_with_map_points(tracking_image, features, map_points);
                    // spdlog::debug("[Viewer] Updated tracking image with {} features", features.size());
                } else {
                    viewer->update_tracking_image(tracking_image);
                    // spdlog::debug("[Viewer] Updated tracking image");
                }
                
                // Update stereo matching image if available
                if (current_frame && current_frame->is_stereo()) {
                    cv::Mat stereo_image = current_frame->draw_stereo_matches();
                    viewer->update_stereo_image(stereo_image);
                    // spdlog::debug("[Viewer] Updated stereo matching image");
                } else {
                    // As fallback, show the left image in right panel too
                    viewer->update_stereo_image(tracking_image);
                    // spdlog::debug("[Viewer] Updated with fallback left image");
                }
                
                previous_frame = current_frame;
            }
            
            // Render viewer
            if (viewer) {
                viewer->render();
            }
        }
        
        // Reset advance_frame flag after processing frame
        if (advance_frame) {
            advance_frame = false;
        }
        
        // Successfully processed this frame
        ++processed_frames;
        
        // Move to next frame
        ++current_idx;
        
        
        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        if(processed_frames % 100 == 0) {
            spdlog::info("[VO] Processed {} / {} frames", processed_frames, end_frame_idx - start_frame_idx);
        }
    }
    
    spdlog::info("[VO] Processing completed! Processed {} frames \n", processed_frames);
    
    // Save trajectories in TUM format for evaluation
    spdlog::info("[TRAJECTORY] Saving trajectories to TUM format...");
    
    // 1. Save estimated trajectory
    std::string estimated_traj_file = dataset_path + "/estimated_trajectory.txt";
    std::ofstream est_file(estimated_traj_file);
    if (est_file.is_open()) {
        const auto& all_frames = estimator.get_all_frames();
        spdlog::info("[TRAJECTORY] Saving {} frames to estimated trajectory", all_frames.size());
        
        for (size_t i = 0; i < all_frames.size(); ++i) {
            const auto& frame = all_frames[i];
            if (!frame) continue;
            
            // Get pose from frame
            Eigen::Matrix4f T_wb = frame->get_Twb();
            
            // Extract translation and rotation
            Eigen::Vector3f translation = T_wb.block<3, 1>(0, 3);
            Eigen::Matrix3f rotation = T_wb.block<3, 3>(0, 0);
            
            // Convert rotation matrix to quaternion (w, x, y, z)
            Eigen::Quaternionf quat(rotation);
            
            // Use matched timestamp for consistency with GT
            if (i < lightweight_vio::EurocUtils::get_matched_count()) {
                long long matched_timestamp = lightweight_vio::EurocUtils::get_matched_timestamp(i);
                double timestamp_sec = static_cast<double>(matched_timestamp) / 1e9;
                
                // TUM format: timestamp tx ty tz qx qy qz qw
                est_file << std::fixed << std::setprecision(6) << timestamp_sec << " "
                         << std::setprecision(8)
                         << translation.x() << " " << translation.y() << " " << translation.z() << " "
                         << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << std::endl;
            }
        }
        est_file.close();
        spdlog::info("[TRAJECTORY] Saved estimated trajectory to: {}", estimated_traj_file);
    } else {
        spdlog::error("[TRAJECTORY] Failed to open estimated trajectory file: {}", estimated_traj_file);
    }
    
    // 2. Save ground truth trajectory
    std::string gt_traj_file = dataset_path + "/ground_truth.txt";
    std::ofstream gt_file(gt_traj_file);
    if (gt_file.is_open()) {
        spdlog::info("[TRAJECTORY] Saving {} ground truth poses", gt_poses.size());
        
        for (size_t i = 0; i < gt_poses.size(); ++i) {
            const auto& gt_pose = gt_poses[i];
            
            // Extract translation and rotation from ground truth
            Eigen::Vector3f translation = gt_pose.block<3, 1>(0, 3);
            Eigen::Matrix3f rotation = gt_pose.block<3, 3>(0, 0);
            
            // Convert rotation matrix to quaternion (w, x, y, z)
            Eigen::Quaternionf quat(rotation);
            
            // Use same timestamp logic as estimated trajectory  
            long long matched_timestamp = lightweight_vio::EurocUtils::get_matched_timestamp(i);
            double timestamp_sec = static_cast<double>(matched_timestamp) / 1e9;
            
            // TUM format: timestamp tx ty tz qx qy qz qw
            gt_file << std::fixed << std::setprecision(6) << timestamp_sec << " "
                    << std::setprecision(8)
                    << translation.x() << " " << translation.y() << " " << translation.z() << " "
                    << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << std::endl;
        }
        gt_file.close();
        spdlog::info("[TRAJECTORY] Saved ground truth trajectory to: {}", gt_traj_file);
    } else {
        spdlog::error("[TRAJECTORY] Failed to open ground truth trajectory file: {}", gt_traj_file);
    }
    
    spdlog::info("[TRAJECTORY] Trajectory files saved successfully! \n\n");
    
    // 3. Frame-to-frame transform comparison
    if (!gt_poses.empty()) {
        
        const auto& all_frames = estimator.get_all_frames();
        std::vector<double> rotation_errors;
        std::vector<double> translation_errors;
        std::vector<double> linear_velocities;  // m/s
        std::vector<double> angular_velocities; // rad/s
        
        for (size_t i = 1; i < all_frames.size() && i < gt_poses.size(); ++i) {
            if (!all_frames[i-1] || !all_frames[i]) continue;
            
            // Calculate estimated transform from frame i-1 to frame i
            Eigen::Matrix4f T_est_prev = all_frames[i-1]->get_Twb();
            Eigen::Matrix4f T_est_curr = all_frames[i]->get_Twb();
            Eigen::Matrix4f T_est_transform = T_est_prev.inverse() * T_est_curr;
            
            // Calculate GT transform from frame i-1 to frame i
            Eigen::Matrix4f T_gt_prev = gt_poses[i-1];
            Eigen::Matrix4f T_gt_curr = gt_poses[i];
            Eigen::Matrix4f T_gt_transform = T_gt_prev.inverse() * T_gt_curr;
            
            // Calculate transform error: T_error = T_gt^-1 * T_est
            Eigen::Matrix4f T_error = T_gt_transform.inverse() * T_est_transform;
            
            // Calculate velocities from estimated transform
            if (i < lightweight_vio::EurocUtils::get_matched_count()) {
                // Get time difference between frames
                long long ts_prev = lightweight_vio::EurocUtils::get_matched_timestamp(i-1);
                long long ts_curr = lightweight_vio::EurocUtils::get_matched_timestamp(i);
                double dt = (ts_curr - ts_prev) / 1e9; // Convert to seconds
                
                if (dt > 0.001 && dt < 1.0) { // Valid time interval
                    // Linear velocity (m/s)
                    Eigen::Vector3f translation = T_est_transform.block<3,1>(0,3);
                    double linear_velocity = translation.norm() / dt;
                    linear_velocities.push_back(linear_velocity);
                    
                    // Angular velocity (rad/s)
                    Eigen::Matrix3f R_est = T_est_transform.block<3,3>(0,0);
                    Eigen::AngleAxisf angle_axis(R_est);
                    double angular_velocity = std::abs(angle_axis.angle()) / dt;
                    angular_velocities.push_back(angular_velocity);
                }
            }
            
            // // Debug: Print detailed info for first few pairs
            // if (i <= 3) {
            //     spdlog::info("=== FRAME PAIR {} -> {} DEBUG (VO) ===", i-1, i);
            //     spdlog::info("GT_prev position: ({:.6f}, {:.6f}, {:.6f})", 
            //                 T_gt_prev(0,3), T_gt_prev(1,3), T_gt_prev(2,3));
            //     spdlog::info("GT_curr position: ({:.6f}, {:.6f}, {:.6f})", 
            //                 T_gt_curr(0,3), T_gt_curr(1,3), T_gt_curr(2,3));
            //     spdlog::info("GT_rel translation: ({:.6f}, {:.6f}, {:.6f})", 
            //                 T_gt_transform(0,3), T_gt_transform(1,3), T_gt_transform(2,3));
            //     spdlog::info("Est_prev position: ({:.6f}, {:.6f}, {:.6f})", 
            //                 T_est_prev(0,3), T_est_prev(1,3), T_est_prev(2,3));
            //     spdlog::info("Est_curr position: ({:.6f}, {:.6f}, {:.6f})", 
            //                 T_est_curr(0,3), T_est_curr(1,3), T_est_curr(2,3));
            //     spdlog::info("Est_rel translation: ({:.6f}, {:.6f}, {:.6f})", 
            //                 T_est_transform(0,3), T_est_transform(1,3), T_est_transform(2,3));
            //     spdlog::info("Error translation: ({:.6f}, {:.6f}, {:.6f})", 
            //                 T_error(0,3), T_error(1,3), T_error(2,3));
            //     spdlog::info("Error magnitude: {:.8f}m", T_error.block<3,1>(0,3).norm());
            //     spdlog::info("========================================");
            // }
            
            // Alternative calculation matching EVO exactly
            // EVO: E_i = (Q_i^-1 * Q_{i+1})^-1 * (P_i^-1 * P_{i+1})
            Eigen::Matrix4f Q_rel = T_gt_prev.inverse() * T_gt_curr;  // GT relative
            Eigen::Matrix4f P_rel = T_est_prev.inverse() * T_est_curr; // Est relative
            Eigen::Matrix4f E_evo = Q_rel.inverse() * P_rel;          // EVO-style error
            
            // Compare our method vs EVO method
            float our_error = T_error.block<3,1>(0,3).norm();
            float evo_error = E_evo.block<3,1>(0,3).norm();
            
            // if (i <= 20) {
            //     spdlog::info("Our method error: {:.8f}m", our_error);
            //     spdlog::info("EVO method error: {:.8f}m", evo_error);
            //     spdlog::info("Difference: {:.8f}m", std::abs(our_error - evo_error));
            // }
            
            // Extract rotation error (angle in degrees)
            Eigen::Matrix3f R_error = T_error.block<3,3>(0,0);
            Eigen::AngleAxisf angle_axis_error(R_error);
            double rotation_error_deg = std::abs(angle_axis_error.angle()) * 180.0 / M_PI;
            
            // Extract translation error (magnitude in meters)
            Eigen::Vector3f t_error = T_error.block<3,1>(0,3);

            // spdlog::debug("Frame {} -> {}: GT_rel=({:.3f},{:.3f},{:.3f}), Est_rel=({:.3f},{:.3f},{:.3f}), Error={:.6f}m", 
            //              i-1, i, 
            //              T_gt_transform(0,3), T_gt_transform(1,3), T_gt_transform(2,3),
            //              T_est_transform(0,3), T_est_transform(1,3), T_est_transform(2,3),
            //              t_error.norm());
            double translation_error_m = t_error.norm();
            
            rotation_errors.push_back(rotation_error_deg);
            translation_errors.push_back(translation_error_m);
            
            
        }
        
        if (!rotation_errors.empty()) {
            // Calculate error statistics
            std::sort(rotation_errors.begin(), rotation_errors.end());
            std::sort(translation_errors.begin(), translation_errors.end());
            
            // Calculate rotation statistics
            double rot_mean = std::accumulate(rotation_errors.begin(), rotation_errors.end(), 0.0) / rotation_errors.size();
            double rot_median = rotation_errors[rotation_errors.size() / 2];
            double rot_min = rotation_errors.front();
            double rot_max = rotation_errors.back();
            double rot_rmse = std::sqrt(std::accumulate(rotation_errors.begin(), rotation_errors.end(), 0.0, 
                [](double sum, double err) { return sum + err * err; }) / rotation_errors.size());
            
            // Calculate translation statistics
            double trans_mean = std::accumulate(translation_errors.begin(), translation_errors.end(), 0.0) / translation_errors.size();
            double trans_median = translation_errors[translation_errors.size() / 2];
            double trans_min = translation_errors.front();
            double trans_max = translation_errors.back();
            double trans_rmse = std::sqrt(std::accumulate(translation_errors.begin(), translation_errors.end(), 0.0,
                [](double sum, double err) { return sum + err * err; }) / translation_errors.size());
            
            // Calculate velocity statistics
            double lin_vel_mean = 0.0, lin_vel_median = 0.0, lin_vel_min = 0.0, lin_vel_max = 0.0;
            double ang_vel_mean = 0.0, ang_vel_median = 0.0, ang_vel_min = 0.0, ang_vel_max = 0.0;
            
            if (!linear_velocities.empty()) {
                std::sort(linear_velocities.begin(), linear_velocities.end());
                lin_vel_mean = std::accumulate(linear_velocities.begin(), linear_velocities.end(), 0.0) / linear_velocities.size();
                lin_vel_median = linear_velocities[linear_velocities.size() / 2];
                lin_vel_min = linear_velocities.front();
                lin_vel_max = linear_velocities.back();
            }
            
            if (!angular_velocities.empty()) {
                std::sort(angular_velocities.begin(), angular_velocities.end());
                ang_vel_mean = std::accumulate(angular_velocities.begin(), angular_velocities.end(), 0.0) / angular_velocities.size();
                ang_vel_median = angular_velocities[angular_velocities.size() / 2];
                ang_vel_min = angular_velocities.front();
                ang_vel_max = angular_velocities.back();
            }
            
            // Beautiful statistics output
            spdlog::info("══════════════════════════════════════════════════════════════════");
            spdlog::info("      FRAME-TO-FRAME TRANSFORM ERROR ANALYSIS (VO) - DEBUG       ");
            spdlog::info("══════════════════════════════════════════════════════════════════");
            spdlog::info(" Total Frame Pairs Analyzed: {} (all_frames: {}, gt_poses: {})", 
                        rotation_errors.size(), all_frames.size(), gt_poses.size());
            spdlog::info(" Frame precision: {} bit floats", sizeof(float) * 8);
            spdlog::info("══════════════════════════════════════════════════════════════════");
            spdlog::info("══════════════════════════════════════════════════════════════════");
            spdlog::info("                    ROTATION ERROR STATISTICS                     ");
            spdlog::info(" Mean      : {:>10.4f}°", rot_mean);
            spdlog::info(" Median    : {:>10.4f}°", rot_median);
            spdlog::info(" Minimum   : {:>10.4f}°", rot_min);
            spdlog::info(" Maximum   : {:>10.4f}°", rot_max);
            spdlog::info(" RMSE      : {:>10.4f}°", rot_rmse);
            spdlog::info("══════════════════════════════════════════════════════════════════");
            spdlog::info("                  TRANSLATION ERROR STATISTICS                    ");
            spdlog::info(" Mean      : {:>10.6f}m", trans_mean);
            spdlog::info(" Median    : {:>10.6f}m", trans_median);
            spdlog::info(" Minimum   : {:>10.6f}m", trans_min);
            spdlog::info(" Maximum   : {:>10.6f}m", trans_max);
            spdlog::info(" RMSE      : {:>10.6f}m", trans_rmse);
            spdlog::info("══════════════════════════════════════════════════════════════════");
            
            // Store transform error statistics for combined output
            transform_stats_available = true;
            transform_rot_mean = rot_mean;
            transform_rot_median = rot_median;
            transform_rot_min = rot_min;
            transform_rot_max = rot_max;
            transform_rot_rmse = rot_rmse;
            transform_trans_mean = trans_mean;
            transform_trans_median = trans_median;
            transform_trans_min = trans_min;
            transform_trans_max = trans_max;
            transform_trans_rmse = trans_rmse;
            transform_total_pairs = rotation_errors.size();
            transform_total_frames = all_frames.size();
            transform_total_gt_poses = gt_poses.size();
            
            // Store velocity statistics
            if (!linear_velocities.empty() && !angular_velocities.empty()) {
                velocity_stats_available = true;
                transform_lin_vel_mean = lin_vel_mean;
                transform_lin_vel_median = lin_vel_median;
                transform_lin_vel_min = lin_vel_min;
                transform_lin_vel_max = lin_vel_max;
                transform_ang_vel_mean = ang_vel_mean;
                transform_ang_vel_median = ang_vel_median;
                transform_ang_vel_min = ang_vel_min;
                transform_ang_vel_max = ang_vel_max;
            }
            
        }
    } else {
        spdlog::warn("[TRANSFORM_ANALYSIS] No ground truth data available for transform analysis");
    }
    
    // 4. Frame processing time - average only
    if (!frame_processing_times.empty()) {
        // Calculate average processing time
        double time_mean = std::accumulate(frame_processing_times.begin(), frame_processing_times.end(), 0.0) / frame_processing_times.size();
        double fps_mean = 1000.0 / time_mean;  // Convert ms to FPS
        
        spdlog::info("[TIMING_ANALYSIS] Average frame processing time: {:.2f}ms ({:.1f}fps)", 
                    time_mean, fps_mean, frame_processing_times.size());

        // Save comprehensive statistics to a single file
        std::string stats_file = dataset_path + "/statistics_vo.txt";
        std::ofstream stats_out(stats_file);
        if (stats_out.is_open()) {
            stats_out << "════════════════════════════════════════════════════════════════════\n";
            stats_out << "                          STATISTICS (VO)                           \n";
            stats_out << "════════════════════════════════════════════════════════════════════\n";
            stats_out << "\n";
            
            // Timing Statistics
            stats_out << "                          TIMING ANALYSIS                           \n";
            stats_out << "════════════════════════════════════════════════════════════════════\n";
            stats_out << " Total Frames Processed: " << frame_processing_times.size() << "\n";
            stats_out << " Average Processing Time: " << std::fixed << std::setprecision(2) << time_mean << "ms\n";
            stats_out << " Average Frame Rate: " << std::fixed << std::setprecision(1) << fps_mean << "fps\n";
            stats_out << "\n";
            
            // Velocity Statistics (if available)
            if (velocity_stats_available) {
                stats_out << "                        VELOCITY STATISTICS                         \n";
                stats_out << "════════════════════════════════════════════════════════════════════\n";
                stats_out << "                     LINEAR VELOCITY STATISTICS                     \n";
                stats_out << " Mean      : " << std::setw(10) << std::fixed << std::setprecision(4) << transform_lin_vel_mean << "m/s\n";
                stats_out << " Median    : " << std::setw(10) << std::fixed << std::setprecision(4) << transform_lin_vel_median << "m/s\n";
                stats_out << " Minimum   : " << std::setw(10) << std::fixed << std::setprecision(4) << transform_lin_vel_min << "m/s\n";
                stats_out << " Maximum   : " << std::setw(10) << std::fixed << std::setprecision(4) << transform_lin_vel_max << "m/s\n";
                stats_out << "\n";
                stats_out << "                    ANGULAR VELOCITY STATISTICS                     \n";
                stats_out << " Mean      : " << std::setw(10) << std::fixed << std::setprecision(4) << transform_ang_vel_mean << "rad/s\n";
                stats_out << " Median    : " << std::setw(10) << std::fixed << std::setprecision(4) << transform_ang_vel_median << "rad/s\n";
                stats_out << " Minimum   : " << std::setw(10) << std::fixed << std::setprecision(4) << transform_ang_vel_min << "rad/s\n";
                stats_out << " Maximum   : " << std::setw(10) << std::fixed << std::setprecision(4) << transform_ang_vel_max << "rad/s\n";
                stats_out << "\n";
            }
            
            // Transform Error Statistics (if available)
            if (transform_stats_available) {
                stats_out << "               FRAME-TO-FRAME TRANSFORM ERROR ANALYSIS              \n";
                stats_out << "════════════════════════════════════════════════════════════════════\n";
                stats_out << " Total Frame Pairs Analyzed: " << transform_total_pairs 
                         << " (all_frames: " << transform_total_frames << ", gt_poses: " << transform_total_gt_poses << ")\n";
                stats_out << " Frame precision: " << sizeof(float) * 8 << " bit floats\n";
                stats_out << "\n";
                stats_out << "                      ROTATION ERROR STATISTICS                     \n";
                stats_out << " Mean      : " << std::setw(10) << std::fixed << std::setprecision(4) << transform_rot_mean << "°\n";
                stats_out << " Median    : " << std::setw(10) << std::fixed << std::setprecision(4) << transform_rot_median << "°\n";
                stats_out << " Minimum   : " << std::setw(10) << std::fixed << std::setprecision(4) << transform_rot_min << "°\n";
                stats_out << " Maximum   : " << std::setw(10) << std::fixed << std::setprecision(4) << transform_rot_max << "°\n";
                stats_out << " RMSE      : " << std::setw(10) << std::fixed << std::setprecision(4) << transform_rot_rmse << "°\n";
                stats_out << "\n";
                stats_out << "                    TRANSLATION ERROR STATISTICS                    \n";
                stats_out << " Mean      : " << std::setw(10) << std::fixed << std::setprecision(6) << transform_trans_mean << "m\n";
                stats_out << " Median    : " << std::setw(10) << std::fixed << std::setprecision(6) << transform_trans_median << "m\n";
                stats_out << " Minimum   : " << std::setw(10) << std::fixed << std::setprecision(6) << transform_trans_min << "m\n";
                stats_out << " Maximum   : " << std::setw(10) << std::fixed << std::setprecision(6) << transform_trans_max << "m\n";
                stats_out << " RMSE      : " << std::setw(10) << std::fixed << std::setprecision(6) << transform_trans_rmse << "m\n";
            } else {
                stats_out << "               FRAME-TO-FRAME TRANSFORM ERROR ANALYSIS              \n";
                stats_out << "════════════════════════════════════════════════════════════════════\n";
                stats_out << " No ground truth data available for transform analysis\n";
            }
            
            stats_out << "\n";
            stats_out << "════════════════════════════════════════════════════════════════════\n";
            stats_out.close();
            spdlog::info("[STATISTICS] Saved comprehensive statistics to: {}", stats_file);
        }

            spdlog::info("════════════════════════════════════════════════════════════════════");
    }
  
    // Wait for user to click Finish button before exiting
    if (viewer) {
        spdlog::info("[VIEWER] Processing completed! Click 'Finish & Exit' button to close the application.");
        while (!viewer->should_close() && !viewer->is_finish_requested()) {
            viewer->render();
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
        spdlog::info("[VIEWER] User requested exit. Shutting down...");
    }
    
    return 0;
}
