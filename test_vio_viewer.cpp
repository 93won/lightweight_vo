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
#include <viewer/ImGuiViewer.h>

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

int main(int argc, char* argv[]) {
    // Initialize spdlog for immediate colored output
    spdlog::set_level(spdlog::level::info);
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
    
    // Load image timestamps
    std::vector<ImageData> image_data = load_image_timestamps(dataset_path);
    if (image_data.empty()) {
        spdlog::error("[Dataset] No images found in dataset");
        return -1;
    }
    
    // Initialize 3D viewer (optional)
    ImGuiViewer* viewer = nullptr;
    std::unique_ptr<ImGuiViewer> viewer_ptr = std::make_unique<ImGuiViewer>();
    if (viewer_ptr->initialize(3840, 1600)) {
        viewer = viewer_ptr.get();
        spdlog::info("[Viewer] 3D viewer initialized successfully");
        
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
            
        auto frame_end = std::chrono::high_resolution_clock::now();
        
        // Timing calculation
        auto preprocess_time = std::chrono::duration_cast<std::chrono::milliseconds>(estimation_start - preprocess_start).count();
        auto estimation_time = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - estimation_start).count();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start).count();
        
        // Update viewer if available
        if (viewer) {
            auto current_frame = estimator.get_current_frame();
            if (current_frame) {
                // Update all accumulated map points
                std::vector<Eigen::Vector3f> all_map_points = extract_all_map_points(estimator);
                viewer->update_points(all_map_points);
                
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
                for (const auto& feature : current_frame->get_features()) {
                    if (feature->has_tracked_feature()) {
                        tracked_features++;
                    } else {
                        new_features++;
                    }
                }
                
                // Update viewer frame information
                viewer->update_frame_info(current_idx + 1, result.num_features, tracked_features, new_features);
                
                viewer->update_tracking_image(tracking_image);
                
                // Update stereo matching image if available
                if (current_frame->is_stereo()) {
                    cv::Mat stereo_image = current_frame->draw_stereo_matches();
                    viewer->update_stereo_image(stereo_image);
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
