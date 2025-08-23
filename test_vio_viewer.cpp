#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <memory>
#include <chrono>
#include <thread>

#include "src/database/Frame.h"
#include "src/database/Feature.h"
#include "src/module/FeatureTracker.h"
#include "src/module/Estimator.h"
#include "src/util/Config.h"
#include "src/viewer/ImGuiViewer.h"

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
    
    // Initialize 3D viewer
    std::cout << "Creating ImGui 3D viewer..." << std::endl;
    ImGuiViewer viewer;
    if (!viewer.initialize(3840, 1600)) {
        std::cerr << "Failed to initialize 3D viewer" << std::endl;
        return -1;
    }
    
    // Initialize Estimator
    Estimator estimator; // Use default constructor
    
    std::shared_ptr<Frame> previous_frame = nullptr;
    int current_idx = 0;
    bool auto_play = true;
    
    std::cout << "Starting VIO estimation with 3D visualization..." << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  3D Viewer: Mouse for camera control" << std::endl;
    std::cout << "  ESC: Exit application" << std::endl;
    
    // Main processing loop
    while (!viewer.shouldClose() && current_idx < image_data.size()) {
        // Load stereo images
        cv::Mat left_image = load_image(dataset_path, image_data[current_idx].filename, 0);
        cv::Mat right_image = load_image(dataset_path, image_data[current_idx].filename, 1);
        
        if (left_image.empty()) {
            current_idx++;
            continue;
        }
        
        // Image preprocessing
        cv::Mat processed_left_image, processed_right_image;
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(left_image, processed_left_image);
        
        if (!right_image.empty()) {
            clahe->apply(right_image, processed_right_image);
        }
        
        // Process frame through estimator
        auto estimation_result = estimator.process_frame(
            processed_left_image, processed_right_image, 
            image_data[current_idx].timestamp);
        
        // Get current frame from estimator
        auto current_frame = estimator.get_current_frame();
        
        // Extract 3D points from current frame
        std::vector<Eigen::Vector3f> current_points = extract_3d_points(current_frame);
        
        // Update 3D viewer with current frame points only (no accumulation)
        viewer.updatePoints(current_points);
        
        // Update images in ImGui viewer
        cv::Mat tracking_image;
        if (previous_frame) {
            tracking_image = current_frame->draw_tracks(*previous_frame);
        } else {
            tracking_image = current_frame->draw_features();
        }
        
        // Add frame information to tracking image
        std::string info = "Frame: " + std::to_string(current_idx + 1) + "/" + 
                          std::to_string(image_data.size()) + 
                          " | Features: " + std::to_string(estimation_result.num_features);
        
        if (current_frame && current_frame->is_stereo()) {
            int stereo_matches = 0;
            int triangulated_points = 0;
            for (const auto& feature : current_frame->get_features()) {
                if (feature->has_stereo_match()) stereo_matches++;
                if (feature->has_3d_point()) triangulated_points++;
            }
            info += " | Stereo: " + std::to_string(stereo_matches);
            info += " | 3D: " + std::to_string(triangulated_points);
        }
        info += " | Current 3D: " + std::to_string(current_points.size());
        if (estimation_result.success) {
            info += " | Estimation: SUCCESS";
        } else {
            info += " | Estimation: FAILED";
        }
        info += " | Time: " + std::to_string(estimation_result.optimization_time_ms) + "ms";
        
        cv::putText(tracking_image, info, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        
        // Update tracking image in viewer
        viewer.updateTrackingImage(tracking_image);
        
        // Update stereo matching image if available
        if (current_frame && current_frame->is_stereo()) {
            cv::Mat stereo_image = current_frame->draw_stereo_matches();
            viewer.updateStereoImage(stereo_image);
        }
        
        // Render 3D viewer
        viewer.render();
        
        // Auto advance
        if (auto_play) {
            current_idx++;
        }
        previous_frame = current_frame;
        
        // Control frame rate
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    
    exit_loop:
    std::cout << "Visualization completed!" << std::endl;
    
    return 0;
}
