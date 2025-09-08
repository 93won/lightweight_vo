/**
 * @file      euroc_stereo_vio.cpp
 * @brief     Main application entry point for the EuRoC stereo VIO pipeline.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-09-08
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

int main(int argc, char* argv[]) {
    // Initialize spdlog for immediate colored output
    spdlog::set_level(spdlog::level::debug);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    
    if (argc != 2) {
        spdlog::error("Usage: {} <euroc_dataset_path>", argv[0]);
        spdlog::error("Example: {} /path/to/MH_01_easy", argv[0]);
        return -1;
    }
    
    // Load configuration
    try {
        Config::getInstance().load("../config/euroc.yaml");
    } catch (const std::exception& e) {
        spdlog::error("[Config] Failed to load configuration: {}", e.what());
        return -1;
    }
    
    std::string dataset_path = argv[1];
    
    // Load EuRoC ground truth data
    if (!lightweight_vio::EurocUtils::load_ground_truth(dataset_path)) {
        spdlog::warn("[EuRoC] Failed to load ground truth data, continuing without it");
    }
    
    // Load IMU data
    if (!lightweight_vio::EurocUtils::load_imu_data(dataset_path)) {
        spdlog::error("[EuRoC] Failed to load IMU data! VIO requires IMU measurements.");
        return -1;
    }
    
    // Print IMU statistics
    lightweight_vio::EurocUtils::print_imu_stats();
    
    // Load image timestamps
    std::vector<ImageData> image_data = load_image_timestamps(dataset_path);
    if (image_data.empty()) {
        spdlog::error("[Dataset] No images found in dataset");
        return -1;
    }
    
    // Pre-match image timestamps with ground truth
    if (lightweight_vio::EurocUtils::has_ground_truth()) {
        std::vector<long long> image_timestamps;
        image_timestamps.reserve(image_data.size());
        for (const auto& img : image_data) {
            image_timestamps.push_back(img.timestamp);
        }
        
        if (lightweight_vio::EurocUtils::match_image_timestamps(image_timestamps)) {
            spdlog::info("[EuRoC] Successfully pre-matched {} image timestamps with ground truth", 
                        lightweight_vio::EurocUtils::get_matched_count());
        }
    }
    
    // Initialize 3D viewer (optional)
    PangolinViewer* viewer = nullptr;
    std::unique_ptr<PangolinViewer> viewer_ptr = std::make_unique<PangolinViewer>();
    if (viewer_ptr->initialize(1920*2, 1080*2)) {
        viewer = viewer_ptr.get();
        spdlog::info("[Viewer] Pangolin viewer initialized successfully");
        
        // Wait for viewer to be fully ready
        spdlog::info("[Viewer] Waiting for viewer to be fully ready...");
        while (viewer && !viewer->is_ready()) {
            viewer->render();
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
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
    
    // Process all frames with IMU data
    size_t current_idx = 0;
    size_t processed_frames = 0;
    long long previous_frame_timestamp = 0;
    
    while (current_idx < image_data.size()) {
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
            viewer->sync_ui_state(auto_play, step_mode);
        }
        
        // In step mode, wait for user input before processing
        if (step_mode && !advance_frame) {
            if (viewer) {
                viewer->render();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            continue;
        }
        
        auto frame_start = std::chrono::high_resolution_clock::now();
        
        // Load stereo images
        cv::Mat left_image = load_image(dataset_path, image_data[current_idx].filename, 0);
        cv::Mat right_image = load_image(dataset_path, image_data[current_idx].filename, 1);
        
        if (left_image.empty()) {
            spdlog::warn("[Dataset] Skipping frame {} due to empty image", current_idx);
            ++current_idx;
            continue;
        }
        
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
        
        // Get IMU data between previous frame and current frame
        std::vector<IMUData> imu_data_from_last_frame;
        
        if (processed_frames > 0) { // Not the first frame
            imu_data_from_last_frame = lightweight_vio::EurocUtils::get_imu_between_timestamps(
                previous_frame_timestamp, image_data[current_idx].timestamp);
        }
        
        // Process frame through estimator with IMU data
        Estimator::EstimationResult result;
        
        if (processed_frames == 0) {
            // First frame - no IMU data needed
            result = estimator.process_frame(
                processed_left_image, processed_right_image, 
                image_data[current_idx].timestamp);
        } else {
            // Use IMU overload for subsequent frames
            result = estimator.process_frame(
                processed_left_image, processed_right_image, 
                image_data[current_idx].timestamp, imu_data_from_last_frame);
        }
        
        // Handle ground truth pose for comparison
        if (lightweight_vio::EurocUtils::get_matched_count() > processed_frames) {
            auto gt_pose_opt = lightweight_vio::EurocUtils::get_matched_pose(processed_frames);
            if (gt_pose_opt.has_value()) {
                Eigen::Matrix4f gt_pose = gt_pose_opt.value();
                gt_poses.push_back(gt_pose);
                
                if (viewer) {
                    viewer->add_ground_truth_pose(gt_pose);
                }
                
                // Log comparison between VIO estimation and GT
                auto current_frame = estimator.get_current_frame();
                if (current_frame) {
                    Eigen::Matrix4f vio_estimated_pose = current_frame->get_Twb();
                    Eigen::Vector3f gt_position = gt_pose.block<3,1>(0,3);
                    Eigen::Vector3f vio_position = vio_estimated_pose.block<3,1>(0,3);
                    float position_error = (gt_position - vio_position).norm();
                }
            }
        }
        
        auto frame_end = std::chrono::high_resolution_clock::now();
        
        // Calculate frame processing time
        auto frame_duration = std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start);
        double frame_time_ms = frame_duration.count() / 1000.0;
        frame_processing_times.push_back(frame_time_ms);
        
        // Update viewer if available
        if (viewer) {
            auto current_frame = estimator.get_current_frame();
            if (current_frame) {
                // Update current pose
                Eigen::Matrix4f current_pose = current_frame->get_Twb();
                viewer->update_pose(current_pose);
                
                // Update current frame camera pose
                Eigen::Matrix4f current_camera_pose = current_frame->get_Twc();
                viewer->update_camera_pose(current_camera_pose);
                
                // Update trajectory
                Eigen::Vector3f current_position = current_pose.block<3, 1>(0, 3);
                static std::vector<Eigen::Vector3f> trajectory_points;
                trajectory_points.push_back(current_position);
                viewer->update_trajectory(trajectory_points);
                
                // Add current frame to viewer
                viewer->add_frame(current_frame);
                
                // Update keyframe window
                const auto keyframes = estimator.get_keyframes_safe();
                viewer->update_keyframe_window(keyframes);
                
                if (!keyframes.empty()) {
                    viewer->set_last_keyframe(keyframes.back());
                    
                    Eigen::Matrix4f last_keyframe_pose = keyframes.back()->get_Twb();
                    Eigen::Matrix4f current_pose_matrix = current_frame->get_Twb();
                    Eigen::Matrix4f relative_pose = last_keyframe_pose.inverse() * current_pose_matrix;
                    viewer->update_relative_pose_from_last_keyframe(relative_pose);
                }
                
                // Update map points
                std::vector<std::shared_ptr<MapPoint>> all_map_points_shared;
                std::vector<std::shared_ptr<MapPoint>> window_map_points_shared;
                
                std::set<std::shared_ptr<MapPoint>> unique_map_points;
                for (const auto& kf : keyframes) {
                    const auto& kf_map_points = kf->get_map_points();
                    for (const auto& mp : kf_map_points) {
                        if (mp && !mp->is_bad()) {
                            unique_map_points.insert(mp);
                        }
                    }
                }
                
                all_map_points_shared.assign(unique_map_points.begin(), unique_map_points.end());
                
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
                
                // Update tracking image
                static std::shared_ptr<Frame> previous_frame = nullptr;
                cv::Mat tracking_image;
                if (previous_frame) {
                    tracking_image = current_frame->draw_tracks(*previous_frame);
                } else {
                    tracking_image = current_frame->draw_features();
                }
                
                // Calculate frame statistics
                int total_features = current_frame->get_feature_count();
                int stereo_matches = 0;
                int map_points_count = 0;
                
                const auto& map_points = current_frame->get_map_points();
                for (const auto& mp : map_points) {
                    if (mp && !mp->is_bad()) {
                        map_points_count++;
                    }
                }
                
                stereo_matches = map_points_count;
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
                    current_idx + 1,
                    total_features,
                    stereo_matches,
                    map_points_count,
                    success_rate,
                    position_error
                );
                
                // Update tracking image with map point indices
                const auto& features = current_frame->get_features();
                const auto& frame_map_points = current_frame->get_map_points();
                viewer->update_tracking_image_with_map_points(tracking_image, features, frame_map_points);
                
                // Update stereo matching image if available
                if (current_frame->is_stereo()) {
                    cv::Mat stereo_image = current_frame->draw_stereo_matches();
                    viewer->update_stereo_image(stereo_image);
                } else {
                    viewer->update_stereo_image(tracking_image);
                }
                
                previous_frame = current_frame;
            }
            
            viewer->render();
        }
        
        // Reset advance_frame flag after processing frame
        if (advance_frame) {
            advance_frame = false;
        }
        
        // Update for next iteration
        previous_frame_timestamp = image_data[current_idx].timestamp;
        ++processed_frames;
        ++current_idx;
        
        // Control frame rate
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
    
    spdlog::info("[VIO] Processing completed! Processed {} frames", processed_frames);
    
    // Save trajectories in TUM format
    spdlog::info("[TRAJECTORY] Saving trajectories to TUM format...");
    
    // Save estimated trajectory
    std::string estimated_traj_file = dataset_path + "estimated_trajectory_vio.txt";
    std::ofstream est_file(estimated_traj_file);
    if (est_file.is_open()) {
        const auto& all_frames = estimator.get_all_frames();
        spdlog::info("[TRAJECTORY] Saving {} frames to estimated VIO trajectory", all_frames.size());
        
        for (const auto& frame : all_frames) {
            if (!frame) continue;
            
            Eigen::Matrix4f T_wb = frame->get_Twb();
            Eigen::Vector3f translation = T_wb.block<3, 1>(0, 3);
            Eigen::Matrix3f rotation = T_wb.block<3, 3>(0, 0);
            Eigen::Quaternionf quat(rotation);
            
            double timestamp_sec = static_cast<double>(frame->get_timestamp()) / 1e9;
            
            est_file << std::fixed << std::setprecision(6) << timestamp_sec << " "
                     << std::setprecision(8)
                     << translation.x() << " " << translation.y() << " " << translation.z() << " "
                     << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << std::endl;
        }
        est_file.close();
        spdlog::info("[TRAJECTORY] Saved estimated VIO trajectory to: {}", estimated_traj_file);
    }
    
    // Save ground truth trajectory
    std::string gt_traj_file = dataset_path + "ground_truth_vio.txt";
    std::ofstream gt_file(gt_traj_file);
    if (gt_file.is_open()) {
        spdlog::info("[TRAJECTORY] Saving {} ground truth poses", gt_poses.size());
        
        for (size_t i = 0; i < gt_poses.size() && i < image_data.size(); ++i) {
            const auto& gt_pose = gt_poses[i];
            
            Eigen::Vector3f translation = gt_pose.block<3, 1>(0, 3);
            Eigen::Matrix3f rotation = gt_pose.block<3, 3>(0, 0);
            Eigen::Quaternionf quat(rotation);
            
            double timestamp_sec = static_cast<double>(image_data[i].timestamp) / 1e9;
            
            gt_file << std::fixed << std::setprecision(6) << timestamp_sec << " "
                    << std::setprecision(8)
                    << translation.x() << " " << translation.y() << " " << translation.z() << " "
                    << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << std::endl;
        }
        gt_file.close();
        spdlog::info("[TRAJECTORY] Saved ground truth trajectory to: {}", gt_traj_file);
    }
    
    spdlog::info("[TRAJECTORY] VIO trajectory files saved successfully!");
    
    // Calculate and display timing statistics
    if (!frame_processing_times.empty()) {
        double time_mean = std::accumulate(frame_processing_times.begin(), frame_processing_times.end(), 0.0) / frame_processing_times.size();
        double fps_mean = 1000.0 / time_mean;
        
        spdlog::info("[TIMING_ANALYSIS] Average frame processing time: {:.2f}ms ({:.1f}fps)", 
                    time_mean, fps_mean);
        spdlog::info("══════════════════════════════════════════════════════════════════");
    }
    
    // Wait for user to click Finish button before exiting
    if (viewer) {
        spdlog::info("[VIEWER] VIO processing completed! Click 'Finish & Exit' button to close the application.");
        while (!viewer->should_close() && !viewer->is_finish_requested()) {
            viewer->render();
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
        spdlog::info("[VIEWER] User requested exit. Shutting down...");
    }
    
    return 0;
}
