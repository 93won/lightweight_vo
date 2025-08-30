#include "EurocUtils.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <optional>
#include <spdlog/spdlog.h>

namespace lightweight_vio {

// Static member definitions
std::vector<EurocUtils::GroundTruthPose> EurocUtils::s_ground_truth_data;
bool EurocUtils::s_data_loaded = false;

// Pre-matched pose data
std::vector<Eigen::Matrix4f> EurocUtils::s_matched_poses;
std::vector<long long> EurocUtils::s_image_timestamps;
std::vector<double> EurocUtils::s_timestamp_errors;

bool EurocUtils::load_ground_truth(const std::string& dataset_root_path) {
    std::string gt_file_path = dataset_root_path + "/mav0/state_groundtruth_estimate0/data.csv";
    
    std::ifstream file(gt_file_path);
    if (!file.is_open()) {
        spdlog::error("[EurocUtils] Failed to open ground truth file: {}", gt_file_path);
        return false;
    }
    
    spdlog::info("[EurocUtils] Loading ground truth from: {}", gt_file_path);
    
    std::string line;
    // Skip header line
    if (!std::getline(file, line)) {
        spdlog::error("[EurocUtils] Ground truth file is empty");
        return false;
    }
    
    s_ground_truth_data.clear();
    
    int line_count = 0;
    while (std::getline(file, line)) {
        line_count++;
        
        std::istringstream iss(line);
        std::string token;
        std::vector<double> values;
        
        // Parse CSV values
        while (std::getline(iss, token, ',')) {
            try {
                values.push_back(std::stod(token));
            } catch (const std::exception& e) {
                spdlog::error("[EurocUtils] Failed to parse value '{}' in line {}: {}", token, line_count, e.what());
                continue;
            }
        }
        
        if (values.size() < 17) {  // timestamp + 3 position + 4 quaternion + 3 velocity + 3 bg + 3 ba
            spdlog::warn("[EurocUtils] Incomplete data in line {}, got {} values, expected 17", line_count, values.size());
            continue;
        }
        
        GroundTruthPose gt_pose;
        
        // Extract data: timestamp, p_RS_R (position), q_RS (quaternion)
        gt_pose.timestamp = static_cast<long long>(values[0]);
        Eigen::Vector3f position(values[1], values[2], values[3]);
        Eigen::Quaternionf quaternion(values[4], values[5], values[6], values[7]);  // w,x,y,z format
        
        // Create 4x4 transformation matrix (T_WS - world to sensor)
        Eigen::Matrix4f T_WS = Eigen::Matrix4f::Identity();
        T_WS.block<3,3>(0,0) = quaternion.toRotationMatrix();
        T_WS.block<3,1>(0,3) = position;

        // T_WS = T_WS.inverse();
        
       
        
        gt_pose.pose = T_WS;
        
        // Extract additional data
        gt_pose.velocity = Eigen::Vector3f(values[8], values[9], values[10]);
        gt_pose.bias_gyro = Eigen::Vector3f(values[11], values[12], values[13]);
        gt_pose.bias_accel = Eigen::Vector3f(values[14], values[15], values[16]);
        
        s_ground_truth_data.push_back(gt_pose);
    }
    
    file.close();
    
    // // Transform all T_WS poses relative to the first T_WS pose (make first T_WS Identity)
    // if (!s_ground_truth_data.empty()) {
    //     // First, extract all T_WS poses and find the first one
    //     std::vector<Eigen::Matrix4f> original_T_WS_poses;
    //     for (const auto& gt_pose : s_ground_truth_data) {
    //         // Reverse the T_SB transformation to get back original T_WS
            
    //         Eigen::Matrix4f T_WS = gt_pose.pose;
    //         original_T_WS_poses.push_back(T_WS);
    //     }
        
    //     // Get inverse of first T_WS
    //     Eigen::Matrix4f first_T_WS_inv = original_T_WS_poses[0].inverse();
        
    
        
    //     for (size_t i = 0; i < s_ground_truth_data.size(); ++i) {
    //         // Make T_WS relative to first T_WS
    //         Eigen::Matrix4f relative_T_WS = original_T_WS_poses[i] * first_T_WS_inv;
            
    //         // Apply T_SB transformation
    //         s_ground_truth_data[i].pose = relative_T_WS;
    //     }
        
    //     spdlog::info("[EurocUtils] Transformed all GT poses relative to first T_WS pose (now Identity)");
    //     spdlog::info("[EurocUtils] First pose after transformation:");
    //     auto& first = s_ground_truth_data[0].pose;
    //     spdlog::info("[EurocUtils]   [{:.6f}, {:.6f}, {:.6f}, {:.6f}]", first(0,0), first(0,1), first(0,2), first(0,3));
    //     spdlog::info("[EurocUtils]   [{:.6f}, {:.6f}, {:.6f}, {:.6f}]", first(1,0), first(1,1), first(1,2), first(1,3));
    //     spdlog::info("[EurocUtils]   [{:.6f}, {:.6f}, {:.6f}, {:.6f}]", first(2,0), first(2,1), first(2,2), first(2,3));
    //     spdlog::info("[EurocUtils]   [{:.6f}, {:.6f}, {:.6f}, {:.6f}]", first(3,0), first(3,1), first(3,2), first(3,3));
    // }
    
    s_data_loaded = true;
    
    spdlog::info("[EurocUtils] Loaded {} ground truth poses", s_ground_truth_data.size());
    print_ground_truth_stats();
    
    return true;
}

Eigen::Matrix4f EurocUtils::get_ground_truth_pose(long long timestamp) {
    if (!s_data_loaded || s_ground_truth_data.empty()) {
        spdlog::warn("[EurocUtils] No ground truth data loaded");
        return Eigen::Matrix4f::Identity();
    }
    
    // Find closest timestamp using binary search
    auto compare = [](const GroundTruthPose& pose, long long ts) {
        return pose.timestamp < ts;
    };
    
    auto it = std::lower_bound(s_ground_truth_data.begin(), s_ground_truth_data.end(), 
                              timestamp, compare);
    
    size_t index = 0;
    if (it == s_ground_truth_data.end()) {
        // Use last pose if timestamp is beyond the last ground truth
        index = s_ground_truth_data.size() - 1;
    } else if (it == s_ground_truth_data.begin()) {
        // Use first pose if timestamp is before the first ground truth
        index = 0;
    } else {
        // Choose between current and previous based on which is closer
        size_t curr_idx = std::distance(s_ground_truth_data.begin(), it);
        size_t prev_idx = curr_idx - 1;
        
        long long curr_diff = std::abs(s_ground_truth_data[curr_idx].timestamp - timestamp);
        long long prev_diff = std::abs(s_ground_truth_data[prev_idx].timestamp - timestamp);
        
        index = (curr_diff < prev_diff) ? curr_idx : prev_idx;
    }
    
    long long time_diff = std::abs(s_ground_truth_data[index].timestamp - timestamp);
    double time_diff_sec = time_diff / 1e9;
    double image_time_sec = timestamp / 1e9;
    double gt_time_sec = s_ground_truth_data[index].timestamp / 1e9;
    
    if (time_diff > 50000000) {  // 50ms threshold
        spdlog::warn("[EurocUtils] Large timestamp difference: {:.3f} sec (image: {:.6f}, GT: {:.6f})", 
                    time_diff_sec, image_time_sec, gt_time_sec);
    } else {
        // Print successful match info for debugging
        spdlog::debug("[EurocUtils] Matched timestamp: image {:.6f} -> GT {:.6f} (diff: {:.6f} sec, index: {})", 
                     image_time_sec, gt_time_sec, time_diff_sec, index);
    }
    
    return s_ground_truth_data[index].pose;
}

Eigen::Matrix4f EurocUtils::get_ground_truth_pose_sec(double timestamp_sec) {
    // Convert seconds to nanoseconds and call the main function
    long long timestamp_ns = static_cast<long long>(timestamp_sec * 1e9);
    return get_ground_truth_pose(timestamp_ns);
}

void EurocUtils::print_ground_truth_stats() {
    if (!s_data_loaded || s_ground_truth_data.empty()) {
        spdlog::info("[EurocUtils] No ground truth data loaded");
        return;
    }
    
    spdlog::info("[EurocUtils] Ground truth statistics:");
    spdlog::info("[EurocUtils]   Total poses: {}", s_ground_truth_data.size());
    
    // Convert timestamps to seconds for readability
    double start_time_sec = s_ground_truth_data.front().timestamp / 1e9;
    double end_time_sec = s_ground_truth_data.back().timestamp / 1e9;
    spdlog::info("[EurocUtils]   Time range: {:.3f} to {:.3f} seconds", start_time_sec, end_time_sec);
    
    double duration_sec = end_time_sec - start_time_sec;
    spdlog::info("[EurocUtils]   Duration: {:.2f} seconds", duration_sec);
    spdlog::info("[EurocUtils]   Average frequency: {:.1f} Hz", s_ground_truth_data.size() / duration_sec);
    
    // Print some timestamp differences to show sync quality
    if (s_ground_truth_data.size() > 1) {
        std::vector<long long> diffs;
        for (size_t i = 1; i < std::min(s_ground_truth_data.size(), size_t(10)); ++i) {
            diffs.push_back(s_ground_truth_data[i].timestamp - s_ground_truth_data[i-1].timestamp);
        }
        
        if (!diffs.empty()) {
            long long avg_diff = 0;
            for (auto diff : diffs) avg_diff += diff;
            avg_diff /= diffs.size();
            
            spdlog::info("[EurocUtils]   Average frame interval: {:.1f} ms", avg_diff / 1e6);
        }
    }
}

bool EurocUtils::has_ground_truth() {
    return s_data_loaded && !s_ground_truth_data.empty();
}

void EurocUtils::clear_ground_truth() {
    s_ground_truth_data.clear();
    s_data_loaded = false;
    
    // Clear matched data as well
    s_matched_poses.clear();
    s_image_timestamps.clear();
    s_timestamp_errors.clear();
    
    spdlog::info("[EurocUtils] Ground truth data cleared");
}

bool EurocUtils::match_image_timestamps(const std::vector<long long>& image_timestamps) {
    if (!s_data_loaded || s_ground_truth_data.empty()) {
        spdlog::error("[EurocUtils] No ground truth data loaded for matching");
        return false;
    }
    
    if (image_timestamps.empty()) {
        spdlog::error("[EurocUtils] No image timestamps provided for matching");
        return false;
    }
    
    spdlog::info("[EurocUtils] Matching {} image timestamps with ground truth data", image_timestamps.size());
    
    s_matched_poses.clear();
    s_image_timestamps.clear();
    s_timestamp_errors.clear();
    
    s_matched_poses.reserve(image_timestamps.size());
    s_image_timestamps.reserve(image_timestamps.size());
    s_timestamp_errors.reserve(image_timestamps.size());
    
    for (size_t i = 0; i < image_timestamps.size(); ++i) {
        long long image_ts = image_timestamps[i];
        
        // Find closest GT timestamp using binary search
        auto compare = [](const GroundTruthPose& pose, long long ts) {
            return pose.timestamp < ts;
        };
        
        auto it = std::lower_bound(s_ground_truth_data.begin(), s_ground_truth_data.end(), 
                                  image_ts, compare);
        
        size_t index = 0;
        if (it == s_ground_truth_data.end()) {
            index = s_ground_truth_data.size() - 1;
        } else if (it == s_ground_truth_data.begin()) {
            index = 0;
        } else {
            size_t curr_idx = std::distance(s_ground_truth_data.begin(), it);
            size_t prev_idx = curr_idx - 1;
            
            long long curr_diff = std::abs(s_ground_truth_data[curr_idx].timestamp - image_ts);
            long long prev_diff = std::abs(s_ground_truth_data[prev_idx].timestamp - image_ts);
            
            index = (curr_diff < prev_diff) ? curr_idx : prev_idx;
        }
        
        // Store matched data
        s_matched_poses.push_back(s_ground_truth_data[index].pose);
        s_image_timestamps.push_back(image_ts);
        
        double time_error_sec = std::abs(s_ground_truth_data[index].timestamp - image_ts) / 1e9;
        s_timestamp_errors.push_back(time_error_sec);
    }
    
    // Print statistics
    double max_error = *std::max_element(s_timestamp_errors.begin(), s_timestamp_errors.end());
    double avg_error = 0.0;
    for (double err : s_timestamp_errors) avg_error += err;
    avg_error /= s_timestamp_errors.size();
    
    spdlog::info("[EurocUtils] Timestamp matching completed:");
    spdlog::info("[EurocUtils]   Matched {} poses", s_matched_poses.size());
    spdlog::info("[EurocUtils]   Average error: {:.6f} sec ({:.1f} ms)", avg_error, avg_error * 1000);
    spdlog::info("[EurocUtils]   Maximum error: {:.6f} sec ({:.1f} ms)", max_error, max_error * 1000);
    
    return true;
}

std::optional<Eigen::Matrix4f> EurocUtils::get_matched_pose(size_t image_index) {
    if (image_index >= s_matched_poses.size()) {
        spdlog::error("[EurocUtils] Invalid image index {} (max: {})", image_index, s_matched_poses.size() - 1);
        return std::nullopt;
    }
    
    return s_matched_poses[image_index];
}

size_t EurocUtils::get_matched_count() {
    return s_matched_poses.size();
}

} // namespace lightweight_vio
