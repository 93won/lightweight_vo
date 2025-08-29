#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <memory>
#include <chrono>
#include <thread>

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
    
    spdlog::info("[Dataset] Loaded {} image timestamps", image_data.size());
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
    
    spdlog::info("[Dataset] Loaded {} image timestamps", image_data.size());
    
    // Initialize 3D viewer
    ImGuiViewer viewer;
    if (!viewer.initialize(3840, 1600)) {
        spdlog::error("[Viewer] Failed to initialize 3D viewer");
        return -1;
    }
    
    // Initialize Estimator
    Estimator estimator; // Use default constructor
    
    std::shared_ptr<Frame> previous_frame = nullptr;
    int current_idx = 0;
    bool auto_play = true;
    bool step_mode = false;
    bool advance_frame = false;
    
    spdlog::info("[VIO] Starting VIO estimation with 3D visualization...");
    spdlog::info("[Controls] 3D Viewer: Mouse for camera control");
    spdlog::info("[Controls] SPACE: Toggle auto-play / step mode");
    spdlog::info("[Controls] N or ENTER: Next frame (in step mode)");
    spdlog::info("[Controls] ESC: Exit application");
    
    // Initialize variables for tracking current state
    Estimator::EstimationResult last_estimation_result;
    std::shared_ptr<Frame> current_frame = nullptr;
    std::vector<Eigen::Vector3f> current_points;
    std::vector<Eigen::Vector3f> all_map_points; // Store all accumulated map points
    std::vector<Eigen::Vector3f> trajectory_points; // Store trajectory
    cv::Mat last_tracking_image;
    
    // Main processing loop
    while (!viewer.shouldClose() && current_idx < image_data.size()) {
        // Check for keyboard input first
        viewer.processKeyboardInput(auto_play, step_mode, advance_frame);
        
        // Determine whether to advance to next frame
        bool should_advance = false;
        if (auto_play && !step_mode) {
            should_advance = true;
        } else if (step_mode && advance_frame) {
            should_advance = true;
            advance_frame = false;  // Reset flag
        }
        
        // Only process new frame if we should advance
        if (should_advance) {
            // Load stereo images
            cv::Mat left_image = load_image(dataset_path, image_data[current_idx].filename, 0);
            cv::Mat right_image = load_image(dataset_path, image_data[current_idx].filename, 1);
            
            if (left_image.empty()) {
                current_idx++;
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
            
            // Process frame through estimator
            last_estimation_result = estimator.process_frame(
                processed_left_image, processed_right_image, 
                image_data[current_idx].timestamp);
            
            // Get current frame from estimator
            current_frame = estimator.get_current_frame();
            
            // Extract 3D points from current frame
            current_points = extract_3d_points(current_frame);
            
            // Update images in ImGui viewer
            if (previous_frame) {
                last_tracking_image = current_frame->draw_tracks(*previous_frame);
            } else {
                last_tracking_image = current_frame->draw_features();
            }
            
            // Add frame information to tracking image (3 lines)
            std::string info1 = "Frame: " + std::to_string(current_idx + 1) + "/" + 
                               std::to_string(image_data.size()) + 
                               " | Features: " + std::to_string(last_estimation_result.num_features);
            
            // Count long-tracked features for all frames
            int long_tracked_points = 0;
            if (current_frame) {
                for (const auto& feature : current_frame->get_features()) {
                    if (feature->get_track_count() >= 3) long_tracked_points++;
                }
            }
            
            std::string info2 = "";
            std::string info3 = "";
            
            if (current_frame && current_frame->is_stereo()) {
                int stereo_matches = 0;
                int triangulated_points = 0;
                for (const auto& feature : current_frame->get_features()) {
                    if (feature->has_stereo_match()) stereo_matches++;
                    if (feature->has_3d_point()) triangulated_points++;
                }
                info2 = "Stereo: " + std::to_string(stereo_matches) + 
                       " | 3D: " + std::to_string(triangulated_points) + 
                       " | Track>=3: " + std::to_string(long_tracked_points);
            } else if (current_frame) {
                info2 = "Track>=3: " + std::to_string(long_tracked_points);
            }
            
            info3 = "Current 3D: " + std::to_string(current_points.size());
            if (last_estimation_result.success) {
                info3 += " | Estimation: SUCCESS";
            } else {
                info3 += " | Estimation: FAILED";
            }
            
            // Add mode information to tracking image
            std::string mode_info = step_mode ? "STEP MODE - Press N/ENTER for next frame" : "AUTO MODE - Press SPACE for step mode";
            cv::putText(last_tracking_image, mode_info, cv::Point(10, 100), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
            
            // Draw three lines of information
            cv::putText(last_tracking_image, info1, cv::Point(10, 25), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            cv::putText(last_tracking_image, info2, cv::Point(10, 50), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            cv::putText(last_tracking_image, info3, cv::Point(10, 75), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            
            previous_frame = current_frame;
            current_idx++;
            
            // Update trajectory only when new frame is processed
            Eigen::Matrix4f current_pose = current_frame->get_Twb();
            Eigen::Vector3f current_position = current_pose.block<3, 1>(0, 3);
            trajectory_points.push_back(current_position);
        }
        
        // Always update viewer with current state (even if no new frame was processed)
        if (current_frame) {
            // Update all accumulated map points from estimator
            all_map_points = extract_all_map_points(estimator);
            viewer.updatePoints(all_map_points);
            
            // Update current pose
            Eigen::Matrix4f current_pose = current_frame->get_Twb();
            viewer.updatePose(current_pose);
            
            // Update trajectory (no longer adding new points here)
            viewer.updateTrajectory(trajectory_points);
            
            // Update tracking image in viewer
            viewer.updateTrackingImage(last_tracking_image);
            
            // Update stereo matching image if available
            if (current_frame->is_stereo()) {
                cv::Mat stereo_image = current_frame->draw_stereo_matches();
                viewer.updateStereoImage(stereo_image);
            }
        }
        
        // Render 3D viewer
        viewer.render();
        
        // Control frame rate
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    
    exit_loop:
    spdlog::info("[VIO] Visualization completed!");
    
    return 0;
}
