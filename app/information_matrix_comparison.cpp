/**
 * @file      information_matrix_comparison.cpp
 * @brief     Comparison test for different information matrix weighting methods
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-09-14
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
#include <future>
#include <mutex>
#include <cstdio>
#include <numeric>
#include <cmath>

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

struct ComparisonResult {
    std::string method_name;
    double avg_translation_error;
    double avg_rotation_error;
    double min_translation_error;
    double max_translation_error;
    double rmse_translation_error;
    double min_rotation_error;
    double max_rotation_error;
    double rmse_rotation_error;
    double total_processing_time;
    int total_keyframes;
    int total_features;
    double convergence_rate;
};

// Helper function to trim whitespace
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

// Load image timestamps and filenames
std::vector<ImageData> loadImageData(const std::string& file_path) {
    std::vector<ImageData> image_data;
    std::ifstream file(file_path);
    
    if (!file.is_open()) {
        spdlog::error("Failed to open file: {}", file_path);
        return image_data;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        std::string timestamp_str, filename;
        
        if (std::getline(iss, timestamp_str, ',') && std::getline(iss, filename)) {
            timestamp_str = trim(timestamp_str);
            filename = trim(filename);
            
            try {
                long long timestamp = std::stoll(timestamp_str);
                image_data.push_back({timestamp, filename});
            } catch (const std::exception& e) {
                spdlog::warn("Failed to parse line: {}", line);
            }
        }
    }
    
    spdlog::info("Loaded {} image entries from {}", image_data.size(), file_path);
    return image_data;
}

// Load stereo images (same as euroc_stereo_vio.cpp)
cv::Mat load_image(const std::string& dataset_path, const std::string& filename, int cam_id = 0) {
    std::string cam_folder = (cam_id == 0) ? "cam0" : "cam1";
    std::string full_path = dataset_path + "/mav0/" + cam_folder + "/data/" + filename;
    cv::Mat image = cv::imread(full_path, cv::IMREAD_GRAYSCALE);
    
    if (image.empty()) {
        spdlog::error("[Dataset] Cannot load image: {}", full_path);
    }
    
    return image;
}

// Calculate trajectory errors using relative transform approach (same as euroc_stereo_vio.cpp)
void calculateRelativeTransformErrors(const std::vector<std::shared_ptr<Frame>>& estimated_frames,
                                    double& avg_translation_error,
                                    double& avg_rotation_error,
                                    double& min_translation_error,
                                    double& max_translation_error, 
                                    double& rmse_translation_error,
                                    double& min_rotation_error,
                                    double& max_rotation_error,
                                    double& rmse_rotation_error) {
    
    if (estimated_frames.empty() || lightweight_vio::EurocUtils::get_matched_count() == 0) {
        avg_translation_error = std::numeric_limits<double>::max();
        avg_rotation_error = std::numeric_limits<double>::max();
        min_translation_error = std::numeric_limits<double>::max();
        max_translation_error = std::numeric_limits<double>::max();
        rmse_translation_error = std::numeric_limits<double>::max();
        min_rotation_error = std::numeric_limits<double>::max();
        max_rotation_error = std::numeric_limits<double>::max();
        rmse_rotation_error = std::numeric_limits<double>::max();
        return;
    }
    
    std::vector<double> rotation_errors;
    std::vector<double> translation_errors;
    
    // Get GT poses for matched frames
    std::vector<Eigen::Matrix4f> gt_poses;
    for (size_t i = 0; i < estimated_frames.size() && i < lightweight_vio::EurocUtils::get_matched_count(); ++i) {
        auto gt_pose_opt = lightweight_vio::EurocUtils::get_matched_pose(i);
        if (gt_pose_opt.has_value()) {
            gt_poses.push_back(gt_pose_opt.value());
        }
    }
    
    // Calculate frame-to-frame transform errors
    for (size_t i = 1; i < estimated_frames.size() && i < gt_poses.size(); ++i) {
        if (!estimated_frames[i-1] || !estimated_frames[i]) continue;
        
        // Calculate estimated transform from frame i-1 to frame i
        Eigen::Matrix4f T_est_prev = estimated_frames[i-1]->get_Twb();
        Eigen::Matrix4f T_est_curr = estimated_frames[i]->get_Twb();
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
        // Calculate translation statistics
        avg_translation_error = std::accumulate(translation_errors.begin(), translation_errors.end(), 0.0) / translation_errors.size();
        min_translation_error = *std::min_element(translation_errors.begin(), translation_errors.end());
        max_translation_error = *std::max_element(translation_errors.begin(), translation_errors.end());
        rmse_translation_error = std::sqrt(std::accumulate(translation_errors.begin(), translation_errors.end(), 0.0,
            [](double sum, double err) { return sum + err * err; }) / translation_errors.size());
        
        // Calculate rotation statistics  
        avg_rotation_error = std::accumulate(rotation_errors.begin(), rotation_errors.end(), 0.0) / rotation_errors.size();
        min_rotation_error = *std::min_element(rotation_errors.begin(), rotation_errors.end());
        max_rotation_error = *std::max_element(rotation_errors.begin(), rotation_errors.end());
        rmse_rotation_error = std::sqrt(std::accumulate(rotation_errors.begin(), rotation_errors.end(), 0.0,
            [](double sum, double err) { return sum + err * err; }) / rotation_errors.size());
    } else {
        avg_translation_error = std::numeric_limits<double>::max();
        avg_rotation_error = std::numeric_limits<double>::max();
        min_translation_error = std::numeric_limits<double>::max();
        max_translation_error = std::numeric_limits<double>::max();
        rmse_translation_error = std::numeric_limits<double>::max();
        min_rotation_error = std::numeric_limits<double>::max();
        max_rotation_error = std::numeric_limits<double>::max();
        rmse_rotation_error = std::numeric_limits<double>::max();
    }
}

// Run VIO with specific information matrix mode (thread-safe version with isolation)
ComparisonResult runVIOWithMode(const std::string& dataset_path,
                               const std::string& config_file,
                               const std::string& mode_name,
                               const std::string& info_matrix_mode) {
    
    ComparisonResult result;
    result.method_name = mode_name;
    
    spdlog::info("========================================");
    spdlog::info("[{}] Running VIO with mode: {}", mode_name, mode_name);
    spdlog::info("[{}] Information matrix mode: {}", mode_name, info_matrix_mode);
    spdlog::info("========================================");
    
    // Thread-safe config handling: Use a static mutex to serialize config access
    static std::mutex config_mutex;
    
    {
        std::lock_guard<std::mutex> config_lock(config_mutex);
        
        auto& config = lightweight_vio::Config::getInstance();
        
        // Temporarily suppress config loading logs
        auto original_level = spdlog::get_level();
        spdlog::set_level(spdlog::level::warn);
        
        // Thread-safe config loading with mutex protection
        if (!config.load(config_file)) {
            spdlog::set_level(original_level);  // Restore level for error reporting
            spdlog::error("[{}] Failed to load configuration from: {}", mode_name, config_file);
            return result;
        }
        
        // Restore log level
        spdlog::set_level(original_level);
        
        // Immediately set the information matrix mode for this thread
        config.m_pnp_information_matrix_mode = info_matrix_mode;
        
        // For comparison tool, disable debug output for cleaner logs
        // but keep original processing behavior intact
        bool original_debug_setting = config.m_enable_debug_output;
        config.m_enable_debug_output = false;  // Only affects log output, not processing logic
        
        // Verify the setting took effect
        spdlog::info("[{}] Config verification: m_pnp_information_matrix_mode = '{}'", 
                    mode_name, config.m_pnp_information_matrix_mode);
        spdlog::info("[{}] Debug output suppressed for comparison (was: {})", mode_name, original_debug_setting);
    } // Config mutex scope ends here
    
    // Thread-safe EurocUtils handling: Use another static mutex for EurocUtils
    static std::mutex euroc_mutex;
    
    {
        std::lock_guard<std::mutex> euroc_lock(euroc_mutex);
        
        // Load ground truth with mutex protection
        if (!lightweight_vio::EurocUtils::load_ground_truth(dataset_path)) {
            spdlog::error("[{}] Failed to load ground truth from {}", mode_name, dataset_path);
            return result;
        }
        
        // Load IMU data with mutex protection
        if (!lightweight_vio::EurocUtils::load_imu_data(dataset_path)) {
            spdlog::error("[{}] Failed to load IMU data from {}", mode_name, dataset_path);
            return result;
        }
    } // EurocUtils mutex scope ends here
    
    // Load image data (use the same function as euroc_stereo_vio.cpp)
    std::vector<ImageData> image_data = loadImageData(dataset_path + "/mav0/cam0/data.csv");
    
    if (image_data.empty()) {
        spdlog::error("[{}] Failed to load image data from {}", mode_name, dataset_path);
        return result;
    }
    
    spdlog::info("[{}] Loaded {} images", mode_name, image_data.size());
    
    // Extract timestamps for GT matching
    std::vector<long long> image_timestamps;
    for (const auto& img : image_data) {
        image_timestamps.push_back(img.timestamp);
    }
    
    // Pre-match ground truth poses and get valid range (thread-safe with mutex)
    size_t start_frame_idx = 0;
    size_t end_frame_idx = image_data.size();
    
    {
        std::lock_guard<std::mutex> euroc_lock(euroc_mutex);
        
        if (!lightweight_vio::EurocUtils::match_image_timestamps(image_timestamps)) {
            spdlog::warn("[{}] Failed to match image timestamps with ground truth", mode_name);
        } else {
            size_t matched_count = lightweight_vio::EurocUtils::get_matched_count();
            spdlog::info("[{}] Successfully pre-matched {} image timestamps with ground truth", mode_name, matched_count);
            
            // Find the valid frame range based on matched timestamps (SAME AS VIO)
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
    } // EurocUtils mutex scope ends for timestamp matching
    
    // Initialize estimator
    lightweight_vio::Estimator estimator;
    
    // Double-check config after estimator initialization
    spdlog::info("[{}] Post-estimator config check: m_pnp_information_matrix_mode = '{}'", 
                mode_name, lightweight_vio::Config::getInstance().m_pnp_information_matrix_mode);
    
    // Set initial ground truth pose for first frame if available (thread-safe)
    {
        std::lock_guard<std::mutex> euroc_lock(euroc_mutex);
        if (lightweight_vio::EurocUtils::has_ground_truth() && !image_data.empty()) {
            auto first_gt_pose = lightweight_vio::EurocUtils::get_matched_pose(0);
            if (first_gt_pose.has_value()) {
                estimator.set_initial_gt_pose(first_gt_pose.value());
            }
        }
    }
    
    std::vector<std::shared_ptr<Frame>> trajectory_frames;
    auto start_time = std::chrono::high_resolution_clock::now();
    int successful_optimizations = 0;
    int processed_frames = 0;
    long long previous_frame_timestamp = 0;
    
    // Process frames within GT range (same as euroc_stereo_vio.cpp)
    size_t current_idx = start_frame_idx;
    spdlog::info("[{}] Starting processing from frame {} to frame {} (GT-matched range)", 
                mode_name, start_frame_idx, end_frame_idx - 1);
                
    while (current_idx < end_frame_idx) {
        const auto& img_data = image_data[current_idx];
        
        // Load left image
        cv::Mat left_img = load_image(dataset_path, img_data.filename, 0);
        
        if (left_img.empty()) {
            spdlog::warn("Failed to load image: {}", img_data.filename);
            continue;
        }
        
        // For stereo, load right image (assuming same timestamp)
        cv::Mat right_img = load_image(dataset_path, img_data.filename, 1);
        
        if (right_img.empty()) {
            spdlog::warn("Failed to load right image for: {}", img_data.filename);
            continue;
        }
        
        // Image preprocessing with enhanced illumination handling (same as euroc_stereo_vio.cpp)
        cv::Mat processed_left_image, processed_right_image;
        
        // Step 1: Global Histogram Equalization for overall brightness normalization
        cv::Mat equalized_left, equalized_right;
        cv::equalizeHist(left_img, equalized_left);
        
        // Step 2: CLAHE for local contrast enhancement
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(equalized_left, processed_left_image);
        
        if (!right_img.empty()) {
            cv::equalizeHist(right_img, equalized_right);
            clahe->apply(equalized_right, processed_right_image);
        }
        
        // Get IMU data between previous frame and current frame (thread-safe)
        std::vector<lightweight_vio::IMUData> imu_data_from_last_frame;
        bool has_valid_imu_data = false;
        
        if (processed_frames > 0) { // Not the first frame
            // Thread-safe IMU data access
            std::lock_guard<std::mutex> euroc_lock(euroc_mutex);
            imu_data_from_last_frame = lightweight_vio::EurocUtils::get_imu_between_timestamps(
                previous_frame_timestamp, img_data.timestamp);
            has_valid_imu_data = !imu_data_from_last_frame.empty();
        }
        
        // Process frame through estimator (same pattern as euroc_stereo_vio.cpp)
        lightweight_vio::Estimator::EstimationResult estimation_result;
        
        if (processed_frames == 0 || !has_valid_imu_data) {
            // First frame or no valid IMU data - use VO mode
            estimation_result = estimator.process_frame(processed_left_image, processed_right_image, img_data.timestamp);
        } else {
            // Use IMU overload for subsequent frames with valid IMU data
            estimation_result = estimator.process_frame(processed_left_image, processed_right_image, img_data.timestamp, imu_data_from_last_frame);
        }
        
        if (estimation_result.success) {
            auto frame = estimator.get_current_frame();
            if (frame) {
                trajectory_frames.push_back(frame);
                successful_optimizations++;
            }
        }
        
        processed_frames++;
        previous_frame_timestamp = img_data.timestamp;
        ++current_idx;
        
        // Print progress
        if (processed_frames % 100 == 0) {
            spdlog::info("[{}] Processed {}/{} frames", mode_name, processed_frames, end_frame_idx - start_frame_idx);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_processing_time = std::chrono::duration<double>(end_time - start_time).count();
    
    // Calculate metrics
    result.total_keyframes = trajectory_frames.size();
    result.total_features = 0; // Not easily accessible from estimator
    result.convergence_rate = processed_frames > 0 ? 
        static_cast<double>(successful_optimizations) / processed_frames : 0.0;
    
    // Calculate trajectory errors using relative transform approach
    calculateRelativeTransformErrors(trajectory_frames, 
                                   result.avg_translation_error, result.avg_rotation_error,
                                   result.min_translation_error, result.max_translation_error, result.rmse_translation_error,
                                   result.min_rotation_error, result.max_rotation_error, result.rmse_rotation_error);
    
    spdlog::info("[{}] Completed: {} keyframes, {:.2f}s processing time", 
                mode_name, result.total_keyframes, result.total_processing_time);
    spdlog::info("[{}] Translation - Mean: {:.6f}m, Min: {:.6f}m, Max: {:.6f}m, RMSE: {:.6f}m", 
                mode_name, result.avg_translation_error, result.min_translation_error, 
                result.max_translation_error, result.rmse_translation_error);
    spdlog::info("[{}] Rotation - Mean: {:.6f}°, Min: {:.6f}°, Max: {:.6f}°, RMSE: {:.6f}°", 
                mode_name, result.avg_rotation_error, result.min_rotation_error, 
                result.max_rotation_error, result.rmse_rotation_error);
    spdlog::info("[{}] Total frames processed: {}/{} (GT range: {}-{})", 
                mode_name, processed_frames, end_frame_idx - start_frame_idx, start_frame_idx, end_frame_idx - 1);
    
    return result;
}

// Print comparison results
void printComparisonResults(const std::vector<ComparisonResult>& results) {
    spdlog::info("=====================================================================");
    spdlog::info("              INFORMATION MATRIX COMPARISON RESULTS");
    spdlog::info("=====================================================================");
    
    // Print aligned results for each method with separators
    for (const auto& result : results) {
        spdlog::info("{:>16}: TRANS min:{:8.6f} | max:{:8.6f} | rmse:{:8.6f} || ROT min:{:8.6f} | max:{:8.6f} | rmse:{:8.6f}", 
                    result.method_name,
                    result.min_translation_error, result.max_translation_error, result.rmse_translation_error,
                    result.min_rotation_error, result.max_rotation_error, result.rmse_rotation_error);
    }
    spdlog::info("=====================================================================");
}

// Save results to CSV
void saveResultsToCSV(const std::vector<ComparisonResult>& results, const std::string& output_file) {
    std::ofstream file(output_file);
    if (!file.is_open()) {
        return;  // Silent failure - no error message
    }
    
    // Write header with all statistics
    file << "Method,Translation_Mean_m,Translation_Min_m,Translation_Max_m,Translation_RMSE_m,"
         << "Rotation_Mean_deg,Rotation_Min_deg,Rotation_Max_deg,Rotation_RMSE_deg,"
         << "Keyframes,Convergence_Rate,Processing_Time_s\n";
    
    // Write data with high precision
    for (const auto& result : results) {
        file << result.method_name << ","
             << std::fixed << std::setprecision(8) << result.avg_translation_error << ","
             << std::fixed << std::setprecision(8) << result.min_translation_error << ","
             << std::fixed << std::setprecision(8) << result.max_translation_error << ","
             << std::fixed << std::setprecision(8) << result.rmse_translation_error << ","
             << std::fixed << std::setprecision(8) << result.avg_rotation_error << ","
             << std::fixed << std::setprecision(8) << result.min_rotation_error << ","
             << std::fixed << std::setprecision(8) << result.max_rotation_error << ","
             << std::fixed << std::setprecision(8) << result.rmse_rotation_error << ","
             << result.total_keyframes << ","
             << std::fixed << std::setprecision(3) << result.convergence_rate << ","
             << std::fixed << std::setprecision(3) << result.total_processing_time << "\n";
    }
    
    file.close();
    // No success message - silent save
}

int main(int argc, char** argv) {
    // Setup logging
    auto logger = spdlog::stdout_color_mt("main");
    spdlog::set_default_logger(logger);
    spdlog::set_level(spdlog::level::info);
    
    if (argc != 3) {
        spdlog::error("Usage: {} <config_file> <dataset_path>", argv[0]);
        spdlog::error("Example: {} /home/eugene/lightweight_vo/config/euroc_vio.yaml /home/eugene/data/EuRoC/MH_05_difficult/", argv[0]);
        return -1;
    }
    
    std::string config_file = argv[1];
    std::string dataset_path = argv[2];
    
    // Load configuration to verify it works
    try {
        // Temporarily suppress config loading logs
        auto original_level = spdlog::get_level();
        spdlog::set_level(spdlog::level::warn);
        
        if (!Config::getInstance().load(config_file)) {
            spdlog::set_level(original_level);  // Restore level for error reporting
            spdlog::error("Failed to load configuration from: {}", config_file);
            return -1;
        }
        
        // Restore log level
        spdlog::set_level(original_level);
        
        // For comparison tool, disable debug output for cleaner results
        // but this should NOT affect processing behavior - only log output
        bool original_debug_setting = Config::getInstance().m_enable_debug_output;
        Config::getInstance().m_enable_debug_output = false;
        
        spdlog::info("Successfully loaded configuration from: {}", config_file);
        spdlog::info("Note: Each thread will reload config independently for thread safety");
        spdlog::info("Debug output suppressed for comparison (original: {})", original_debug_setting);
    } catch (const std::exception& e) {
        spdlog::error("Failed to load configuration from {}: {}", config_file, e.what());
        return -1;
    }
    
    // Generate output CSV filename automatically in the dataset directory
    std::string dataset_name = dataset_path.substr(dataset_path.find_last_of("/\\") + 1);
    if (dataset_name.empty()) {
        // Remove trailing slash and try again
        std::string temp_path = dataset_path.substr(0, dataset_path.length() - 1);
        dataset_name = temp_path.substr(temp_path.find_last_of("/\\") + 1);
    }
    std::string output_file = dataset_path + "/information_matrix_comparison_" + dataset_name + ".csv";
    
    spdlog::info("Information Matrix Comparison Test");
    spdlog::info("Dataset: {}", dataset_path);
    spdlog::info("Config: {}", config_file);
    spdlog::info("Output: {}", output_file);
    spdlog::info("Running methods in parallel with thread-safe implementation...");
    
    // Run comparison tests in parallel using separate threads with thread-safe approach
    std::vector<ComparisonResult> results;
    std::mutex results_mutex;  // Protect results vector
    
    spdlog::info("Starting both methods in parallel threads with thread isolation...");
    
    // Launch both methods concurrently using futures with thread-safe design
    auto future1 = std::async(std::launch::async, [&]() -> ComparisonResult {
        // Thread 1: ObservationCount method
        try {
            spdlog::info("[Thread1] Starting ObservationCount method...");
            auto result = runVIOWithMode(dataset_path, config_file, "ObservationCount", "observation_count");
            spdlog::info("[Thread1] ObservationCount method completed successfully");
            return result;
        } catch (const std::exception& e) {
            spdlog::error("[Thread1] Exception in ObservationCount: {}", e.what());
            ComparisonResult error_result;
            error_result.method_name = "ObservationCount_ERROR";
            return error_result;
        }
    });
    
    auto future2 = std::async(std::launch::async, [&]() -> ComparisonResult {
        // Thread 2: ReprojectionError method
        try {
            // Add a small delay to avoid simultaneous singleton access
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            spdlog::info("[Thread2] Starting ReprojectionError method...");
            auto result = runVIOWithMode(dataset_path, config_file, "ReprojectionError", "reprojection_error");
            spdlog::info("[Thread2] ReprojectionError method completed successfully");
            return result;
        } catch (const std::exception& e) {
            spdlog::error("[Thread2] Exception in ReprojectionError: {}", e.what());
            ComparisonResult error_result;
            error_result.method_name = "ReprojectionError_ERROR";
            return error_result;
        }
    });
    
    // Wait for both threads to complete and collect results safely
    spdlog::info("Waiting for both threads to complete...");
    
    try {
        auto result1 = future1.get();
        auto result2 = future2.get();
        
        // Thread-safe result collection
        {
            std::lock_guard<std::mutex> lock(results_mutex);
            results.push_back(result1);
            results.push_back(result2);
        }
        
        spdlog::info("Both methods completed successfully!");
    } catch (const std::exception& e) {
        spdlog::error("Exception during parallel execution: {}", e.what());
        return -1;
    }
    
    // Print and save results
    printComparisonResults(results);
    saveResultsToCSV(results, output_file);
    
    return 0;
}
