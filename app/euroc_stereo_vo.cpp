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
        spdlog::warn("[MAP_POINTS] No current frame available");
        return points;
    }
    
    const auto& frame_map_points = current_frame->get_map_points();
    // spdlog::info("[MAP_POINTS] Current frame has {} map point slots", frame_map_points.size());
    
    int valid_count = 0;
    for (const auto& mp : frame_map_points) {
        if (mp && !mp->is_bad()) {
            Eigen::Vector3f position = mp->get_position();
            points.push_back(position);
            valid_count++;
        }
    }
    
    // spdlog::info("[MAP_POINTS] Found {} valid map points in current frame", valid_count);
    
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
    
    // Process all frames
    size_t current_idx = 0;
    size_t processed_frames = 0;
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
        if (lightweight_vio::EurocUtils::get_matched_count() > processed_frames) {
            auto gt_pose_opt = lightweight_vio::EurocUtils::get_matched_pose(processed_frames);
            if (gt_pose_opt.has_value()) {
                Eigen::Matrix4f gt_pose = gt_pose_opt.value();
                
                // Store ground truth pose for trajectory export
                gt_poses.push_back(gt_pose);
                
                // For the first frame only: apply GT pose to initialize VIO with same starting point
                // TEMPORARILY DISABLED: Let VIO start from identity to avoid coordinate conflicts
                if (false && processed_frames == 0) {
                    estimator.apply_gt_pose_to_current_frame(gt_pose);
                    // spdlog::info("[GT_INIT] First frame initialized with GT pose for fair comparison");
                } else if (processed_frames == 0) {
                    // spdlog::info("[VIO_INIT] First frame initialized with identity pose (no GT applied)");
                }
                
                // Always add GT pose to viewer trajectory for comparison
                if (viewer) {
                    viewer->add_ground_truth_pose(gt_pose);
                }
                
                // Log comparison between VIO estimation and GT
                auto current_frame = estimator.get_current_frame();
                if (current_frame) {
                    Eigen::Matrix4f vio_estimated_pose = current_frame->get_Twb();
                    
                    // Extract positions for comparison
                    Eigen::Vector3f gt_position = gt_pose.block<3,1>(0,3);
                    Eigen::Vector3f vio_position = vio_estimated_pose.block<3,1>(0,3);
                    
                    float position_error = (gt_position - vio_position).norm();
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
                
                // Update trajectory
                Eigen::Vector3f current_position = current_pose.block<3, 1>(0, 3);
                static std::vector<Eigen::Vector3f> trajectory_points;
                trajectory_points.push_back(current_position);
                viewer->update_trajectory(trajectory_points);
                
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
        
        // Control frame rate
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
    
    spdlog::info("[VO] Processing completed! Processed {} frames \n", processed_frames);
    
    // Save trajectories in TUM format for evaluation
    spdlog::info("[TRAJECTORY] Saving trajectories to TUM format...");
    
    // 1. Save estimated trajectory
    std::string estimated_traj_file = dataset_path + "estimated_trajectory.txt";
    std::ofstream est_file(estimated_traj_file);
    if (est_file.is_open()) {
        const auto& all_frames = estimator.get_all_frames();
        spdlog::info("[TRAJECTORY] Saving {} frames to estimated trajectory", all_frames.size());
        
        for (const auto& frame : all_frames) {
            if (!frame) continue;
            
            // Get pose from frame
            Eigen::Matrix4f T_wb = frame->get_Twb();
            
            // Extract translation and rotation
            Eigen::Vector3f translation = T_wb.block<3, 1>(0, 3);
            Eigen::Matrix3f rotation = T_wb.block<3, 3>(0, 0);
            
            // Convert rotation matrix to quaternion (w, x, y, z)
            Eigen::Quaternionf quat(rotation);
            
            // Convert timestamp from nanoseconds to seconds
            double timestamp_sec = static_cast<double>(frame->get_timestamp()) / 1e9;
            
            // TUM format: timestamp tx ty tz qx qy qz qw
            est_file << std::fixed << std::setprecision(6) << timestamp_sec << " "
                     << std::setprecision(8)
                     << translation.x() << " " << translation.y() << " " << translation.z() << " "
                     << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << std::endl;
        }
        est_file.close();
        spdlog::info("[TRAJECTORY] Saved estimated trajectory to: {}", estimated_traj_file);
    } else {
        spdlog::error("[TRAJECTORY] Failed to open estimated trajectory file: {}", estimated_traj_file);
    }
    
    // 2. Save ground truth trajectory
    std::string gt_traj_file = dataset_path + "ground_truth.txt";
    std::ofstream gt_file(gt_traj_file);
    if (gt_file.is_open()) {
        spdlog::info("[TRAJECTORY] Saving {} ground truth poses", gt_poses.size());
        
        for (size_t i = 0; i < gt_poses.size() && i < image_data.size(); ++i) {
            const auto& gt_pose = gt_poses[i];
            
            // Extract translation and rotation from ground truth
            Eigen::Vector3f translation = gt_pose.block<3, 1>(0, 3);
            Eigen::Matrix3f rotation = gt_pose.block<3, 3>(0, 0);
            
            // Convert rotation matrix to quaternion (w, x, y, z)
            Eigen::Quaternionf quat(rotation);
            
            // Convert timestamp from nanoseconds to seconds
            double timestamp_sec = static_cast<double>(image_data[i].timestamp) / 1e9;
            
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
            
            // Extract rotation error (angle in degrees)
            Eigen::Matrix3f R_error = T_error.block<3,3>(0,0);
            Eigen::AngleAxisf angle_axis(R_error);
            double rotation_error_deg = std::abs(angle_axis.angle()) * 180.0 / M_PI;
            
            // Extract translation error (magnitude in meters)
            Eigen::Vector3f t_error = T_error.block<3,1>(0,3);
            double translation_error_m = t_error.norm();
            
            rotation_errors.push_back(rotation_error_deg);
            translation_errors.push_back(translation_error_m);
            
            
        }
        
        if (!rotation_errors.empty()) {
            // Calculate statistics
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
            
            // Beautiful statistics output
            spdlog::info("══════════════════════════════════════════════════════════════════");
            spdlog::info("            FRAME-TO-FRAME TRANSFORM ERROR ANALYSIS              ");
            spdlog::info("══════════════════════════════════════════════════════════════════");
            spdlog::info(" Total Frame Pairs Analyzed: {}", rotation_errors.size());
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

            spdlog::info("══════════════════════════════════════════════════════════════════");
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
