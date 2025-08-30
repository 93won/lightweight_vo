#pragma once

#include <vector>
#include <string>
#include <optional>
#include <Eigen/Dense>

namespace lightweight_vio {

/**
 * @brief Utility class for EuRoC dataset specific operations
 * Handles ground truth data loading and timestamp synchronization
 */
class EurocUtils {
public:
    struct GroundTruthPose {
        long long timestamp;
        Eigen::Matrix4f pose;
        Eigen::Vector3f velocity;
        Eigen::Vector3f bias_gyro;
        Eigen::Vector3f bias_accel;
    };

    /**
     * @brief Load ground truth data from EuRoC dataset
     * @param dataset_root_path Root path of the EuRoC dataset (e.g., /path/to/V1_01_easy)
     * @return True if loaded successfully
     */
    static bool load_ground_truth(const std::string& dataset_root_path);

    /**
     * @brief Get ground truth pose for a given timestamp
     * @param timestamp Timestamp in nanoseconds
     * @return Ground truth pose matrix (4x4)
     */
    static Eigen::Matrix4f get_ground_truth_pose(long long timestamp);

    /**
     * @brief Get ground truth pose for a given timestamp in seconds
     * @param timestamp_sec Timestamp in seconds
     * @return Ground truth pose matrix (4x4)
     */
    static Eigen::Matrix4f get_ground_truth_pose_sec(double timestamp_sec);

    /**
     * @brief Pre-match ground truth poses for given image timestamps
     * @param image_timestamps Vector of image timestamps in nanoseconds
     * @return True if matching was successful
     */
    static bool match_image_timestamps(const std::vector<long long>& image_timestamps);

    /**
     * @brief Get pre-matched ground truth pose by image index
     * @param image_index Index of the image (0-based)
     * @return Ground truth pose matrix (4x4)
     */
    static std::optional<Eigen::Matrix4f> get_matched_pose(size_t image_index);

    /**
     * @brief Get number of matched poses
     * @return Number of pre-matched poses
     */
    static size_t get_matched_count();

    /**
     * @brief Print statistics about loaded ground truth data
     */
    static void print_ground_truth_stats();

    /**
     * @brief Check if ground truth data is loaded
     * @return True if data is available
     */
    static bool has_ground_truth();

    /**
     * @brief Clear all loaded ground truth data
     */
    static void clear_ground_truth();

private:
    static std::vector<GroundTruthPose> s_ground_truth_data;
    static bool s_data_loaded;
    
    // Pre-matched poses for image timestamps
    static std::vector<Eigen::Matrix4f> s_matched_poses;
    static std::vector<long long> s_image_timestamps;
    static std::vector<double> s_timestamp_errors;  // In seconds
};

} // namespace lightweight_vio
