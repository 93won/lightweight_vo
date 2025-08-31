#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <iostream>
#include <fstream>
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
#include <module/FeatureTracker.h>
#include <module/Estimator.h>
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
    spdlog::info("[MAP_POINTS] Current frame has {} map point slots", frame_map_points.size());
    
    int valid_count = 0;
    for (const auto& mp : frame_map_points) {
        if (mp && !mp->is_bad()) {
            Eigen::Vector3f position = mp->get_position();
            points.push_back(position);
            valid_count++;
        }
    }
    
    spdlog::info("[MAP_POINTS] Found {} valid map points in current frame", valid_count);
    
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
    
    spdlog::debug("[ALL_MAP_POINTS] Found {} total valid map points", valid_count);
    
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
    if (viewer_ptr->initialize(3200, 2400)) {  // Pangolin 뷰어 초기화
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
    
    // Process all frames
    size_t current_idx = 0;
    size_t processed_frames = 0;
    while (current_idx < image_data.size()) {
        // Check if viewer wants to exit
        if (viewer && viewer->should_close()) {
            spdlog::info("[Viewer] User requested exit");
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
                    
                    spdlog::info("[GT_COMPARE] Frame {}: GT ({:.3f}, {:.3f}, {:.3f}), VIO ({:.3f}, {:.3f}, {:.3f}), Error: {:.3f}m",
                               processed_frames,
                               gt_position.x(), gt_position.y(), gt_position.z(),
                               vio_position.x(), vio_position.y(), vio_position.z(),
                               position_error);
                }
            }
        }
            
        auto frame_end = std::chrono::high_resolution_clock::now();
        
        // Timing calculation
        auto preprocess_time = std::chrono::duration_cast<std::chrono::milliseconds>(estimation_start - preprocess_start).count();
        auto estimation_time = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - estimation_start).count();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start).count();
        
        // Update viewer if available
        if (viewer) {
            auto current_frame = estimator.get_current_frame();
            if (current_frame) {
                // Get both accumulated and current frame map points
                std::vector<Eigen::Vector3f> all_map_points = extract_all_accumulated_map_points(estimator);
                std::vector<Eigen::Vector3f> current_map_points = extract_current_frame_map_points(estimator);
                
                // Update map points with color differentiation
                viewer->update_map_points(all_map_points, current_map_points);
                
                // Update current pose
                Eigen::Matrix4f current_pose = current_frame->get_Twb();
                viewer->update_pose(current_pose);
                
                // Update trajectory
                Eigen::Vector3f current_position = current_pose.block<3, 1>(0, 3);
                static std::vector<Eigen::Vector3f> trajectory_points;
                trajectory_points.push_back(current_position);
                viewer->update_trajectory(trajectory_points);
                
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
                float success_rate = (result.num_features > 0) ? 
                    (static_cast<float>(stereo_matches) / static_cast<float>(result.num_features)) * 100.0f : 0.0f;
                
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
                    result.num_features,       // total_features  
                    stereo_matches,            // stereo_matches
                    map_points_count,          // map_points
                    success_rate,              // success_rate
                    position_error             // position_error
                );
                
                // Update viewer frame information (legacy)
                viewer->update_frame_info(current_idx + 1, result.num_features, tracked_features, new_features);
                
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
                    // For testing: show processed right image if available
                    if (!processed_right_image.empty()) {
                        viewer->update_stereo_image(processed_right_image);
                        // spdlog::debug("[Viewer] Updated with right image");
                    } else {
                        // As fallback, show the left image in right panel too
                        viewer->update_stereo_image(tracking_image);
                        // spdlog::debug("[Viewer] Updated with fallback left image");
                    }
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
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    
    spdlog::info("[VIO] Processing completed! Processed {} frames", processed_frames);
    
    return 0;
}
