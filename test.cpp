#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <memory>
#include <chrono>

#include "src/database/Frame.h"
#include "src/database/Feature.h"
#include "src/module/FeatureTracker.h"
#include "src/util/Config.h"

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
        std::cerr << "Cannot open data.csv file: " << data_file << std::endl;
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
    
    std::cout << "Loaded " << image_data.size() << " image timestamps" << std::endl;
    return image_data;
}

cv::Mat load_image(const std::string& dataset_path, const std::string& filename, int cam_id = 0) {
    std::string cam_folder = (cam_id == 0) ? "cam0" : "cam1";
    std::string full_path = dataset_path + "/mav0/" + cam_folder + "/data/" + filename;
    cv::Mat image = cv::imread(full_path, cv::IMREAD_GRAYSCALE);
    
    if (image.empty()) {
        std::cerr << "Cannot load image: " << full_path << std::endl;
    }
    
    return image;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <euroc_dataset_path>" << std::endl;
        std::cerr << "Example: " << argv[0] << " /path/to/MH_01_easy" << std::endl;
        return -1;
    }
    
    // Load configuration
    try {
        Config::getInstance().load("../config/euroc.yaml");
        std::cout << "Configuration loaded successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load configuration: " << e.what() << std::endl;
        return -1;
    }
    
    std::string dataset_path = argv[1];
    std::cout << "Loading EuRoC dataset from: " << dataset_path << std::endl;
    
    // Load image timestamps
    std::vector<ImageData> image_data = load_image_timestamps(dataset_path);
    if (image_data.empty()) {
        std::cerr << "No images found in dataset" << std::endl;
        return -1;
    }
    
    // Initialize feature tracker
    FeatureTracker tracker;
    
    std::shared_ptr<Frame> previous_frame = nullptr;
    int frame_id = 0;
    int current_idx = 0;
    bool auto_play = false;
    bool show_stereo_view = false;  // Toggle for stereo visualization
    
    std::cout << "Starting feature tracking..." << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  'q' or ESC: quit" << std::endl;
    std::cout << "  'space': next image" << std::endl;
    std::cout << "  'p': previous image" << std::endl;
    std::cout << "  'a': auto play toggle" << std::endl;
    std::cout << "  'r': reset to first image" << std::endl;
    std::cout << "  's': toggle stereo matching view" << std::endl;
    
    cv::namedWindow("Lightweight VIO - Feature Tracking", cv::WINDOW_AUTOSIZE);
    
    while (true) {
        if (current_idx < 0) current_idx = 0;
        if (current_idx >= image_data.size()) {
            if (auto_play) {
                current_idx = 0; // Loop back to beginning
            } else {
                current_idx = image_data.size() - 1;
            }
        }
        
        // Load stereo images
        cv::Mat left_image = load_image(dataset_path, image_data[current_idx].filename, 0);
        cv::Mat right_image = load_image(dataset_path, image_data[current_idx].filename, 1);
        
        if (left_image.empty()) {
            current_idx++;
            continue;
        }
        
        // Image preprocessing for left image
        cv::Mat processed_left_image;
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(left_image, processed_left_image);
        
        // Image preprocessing for right image (if available)
        cv::Mat processed_right_image;
        if (!right_image.empty()) {
            clahe->apply(right_image, processed_right_image);
        }
        
        // Create current frame with stereo images
        auto frame_start = std::chrono::high_resolution_clock::now();
        
        auto current_frame = std::make_shared<Frame>(image_data[current_idx].timestamp, current_idx);
        if (!processed_right_image.empty()) {
            current_frame->set_stereo_images(processed_left_image, processed_right_image);
        } else {
            current_frame->set_left_image(processed_left_image);
        }
        
        // Track features
        tracker.track_features(current_frame, previous_frame);
        
        // Compute stereo matches if stereo data is available
        if (current_frame->is_stereo()) {
            current_frame->compute_stereo_matches();
            
            // Undistort feature coordinates
            current_frame->undistort_features();
            
            // Triangulate 3D points from stereo matches
            current_frame->triangulate_stereo_points();
        }
        
        auto frame_end = std::chrono::high_resolution_clock::now();
        auto frame_duration = std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start);
        
        std::cout << "[TIMING] ==== TOTAL FRAME PROCESSING: " << frame_duration.count() / 1000.0 << " ms ====" << std::endl;
        std::cout << std::endl;  // Add blank line for readability
        
        // Draw features and tracks
        cv::Mat display_image;
        if (show_stereo_view && current_frame->is_stereo()) {
            // Show stereo matching visualization
            display_image = current_frame->draw_stereo_matches();
        } else if (previous_frame) {
            display_image = current_frame->draw_tracks(*previous_frame);
        } else {
            display_image = current_frame->draw_features();
        }
        
        // Add frame information
        std::string info = "Frame: " + std::to_string(current_idx + 1) + "/" + 
                          std::to_string(image_data.size()) + 
                          " | Features: " + std::to_string(current_frame->get_feature_count());
        if (current_frame->is_stereo()) {
            // Count stereo matches and 3D points
            int stereo_matches = 0;
            int triangulated_points = 0;
            for (const auto& feature : current_frame->get_features()) {
                if (feature->has_stereo_match()) stereo_matches++;
                if (feature->has_3d_point()) triangulated_points++;
            }
            info += " | Stereo: " + std::to_string(stereo_matches);
            info += " | 3D: " + std::to_string(triangulated_points);
        }
        info += " | TS: " + std::to_string(image_data[current_idx].timestamp);
        
        cv::putText(display_image, info, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        
        if (auto_play) {
            cv::putText(display_image, "AUTO PLAY", cv::Point(10, 60), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        }
        
        if (show_stereo_view && current_frame->is_stereo()) {
            cv::putText(display_image, "STEREO VIEW", cv::Point(10, 90), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 255), 2);
        }
        
        // Show image
        cv::imshow("Lightweight VIO - Feature Tracking", display_image);
        
        // Handle keyboard input
        int key;
        if (auto_play) {
            key = cv::waitKey(50); // 50ms delay for auto play
            if (key == -1) { // No key pressed
                current_idx++;
                previous_frame = current_frame;
                continue;
            }
        } else {
            key = cv::waitKey(0);
        }
        
        switch (key & 0xFF) {
            case 'q':
            case 27: // ESC
                goto exit_loop;
            case ' ': // Space - next image
                current_idx++;
                break;
            case 'p': // Previous image
                current_idx--;
                // Reset when going backwards
                previous_frame = nullptr;
                break;
            case 'a': // Auto play toggle
                auto_play = !auto_play;
                std::cout << "Auto play: " << (auto_play ? "ON" : "OFF") << std::endl;
                break;
            case 'r': // Reset to first image
                current_idx = 0;
                previous_frame = nullptr;
                break;
            case 's': // Toggle stereo view
                show_stereo_view = !show_stereo_view;
                std::cout << "Stereo view: " << (show_stereo_view ? "ON" : "OFF") << std::endl;
                break;
            default:
                break;
        }
        
        // Update previous frame
        previous_frame = current_frame;
    }
    
exit_loop:
    cv::destroyAllWindows();
    std::cout << "Feature tracking completed!" << std::endl;
    
    return 0;
}
