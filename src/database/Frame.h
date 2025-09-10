/**
 * @file      Frame.h
 * @brief     Defines the Frame class, representing a single camera capture.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-08-11
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <vector>
#include <map>
#include <memory>
#include <chrono>
#include <unordered_map>
#include <mutex>

namespace lightweight_vio {

class Feature; // Forward declaration
class MapPoint;
struct IMUPreintegration; // Forward declaration for IMU preintegration

// IMU measurement structure
struct IMUData {
    double timestamp;              // timestamp in seconds
    Eigen::Vector3f linear_accel;  // linear acceleration [m/s^2]
    Eigen::Vector3f angular_vel;   // angular velocity [rad/s]
    
    IMUData() = default;
    IMUData(double ts, const Eigen::Vector3f& accel, const Eigen::Vector3f& gyro)
        : timestamp(ts), linear_accel(accel), angular_vel(gyro) {}
};

class Frame {
public:
    // Constructors
    Frame(long long timestamp, int frame_id);
    Frame(long long timestamp, int frame_id, 
          double fx, double fy, double cx, double cy, 
          const std::vector<double>& distortion_coeffs);
    
    // Stereo constructor - directly takes both images with manual camera params
    Frame(long long timestamp, int frame_id,
          const cv::Mat& left_image, const cv::Mat& right_image,
          double fx, double fy, double cx, double cy, 
          const std::vector<double>& distortion_coeffs);
          
    // Simple stereo constructor - uses Config for camera parameters
    Frame(long long timestamp, int frame_id,
          const cv::Mat& left_image, const cv::Mat& right_image);
    ~Frame(); // Use explicit destructor


    // Getters
    long long get_timestamp() const { return m_timestamp; }
    int get_frame_id() const { return m_frame_id; }
    const cv::Mat& get_left_image() const { return m_left_image; }
    const cv::Mat& get_right_image() const { return m_right_image; }
    const cv::Mat& get_image() const { return m_left_image; } // For backward compatibility
    const std::vector<std::shared_ptr<Feature>>& get_features() const { return m_features; }
    std::vector<std::shared_ptr<Feature>>& get_features_mutable() { return m_features; }
    const Eigen::Matrix3f& get_rotation() const { return m_rotation; }
    const Eigen::Vector3f& get_translation() const { return m_translation; }
    bool is_keyframe() const { return m_is_keyframe; }
    bool is_stereo() const { return true; } // Always stereo

    // Pose management
    void set_pose(const Eigen::Matrix3f& rotation, const Eigen::Vector3f& translation);
    void set_Twb(const Eigen::Matrix4f& T_wb);
    Eigen::Matrix4f get_Twb() const;
    Eigen::Matrix4f get_Twc() const;  // World to camera transform
    
    // VIO-specific getters (for optimization)
    Sophus::SE3f get_world_pose() const;  // Get SE3 world pose
    void set_world_pose(const Sophus::SE3f& pose);  // Set SE3 world pose
    Eigen::Vector3f get_velocity() const { return m_velocity; }  // Get velocity
    void set_velocity(const Eigen::Vector3f& velocity) { m_velocity = velocity; }  // Set velocity
    
    // Auto velocity initialization from preintegration
    void initialize_velocity_from_preintegration();  // Initialize velocity from stored preintegration data
    
    // IMU bias management
    Eigen::Vector3f get_accel_bias() const { return m_accel_bias; }  // Get accelerometer bias
    void set_accel_bias(const Eigen::Vector3f& accel_bias) { m_accel_bias = accel_bias; }  // Set accelerometer bias
    Eigen::Vector3f get_gyro_bias() const { return m_gyro_bias; }  // Get gyroscope bias
    void set_gyro_bias(const Eigen::Vector3f& gyro_bias) { m_gyro_bias = gyro_bias; }  // Set gyroscope bias
    
    // IMU time difference from last keyframe
    double get_dt_from_last_keyframe() const { return m_dt_from_last_keyframe; }
    void set_dt_from_last_keyframe(double dt) { m_dt_from_last_keyframe = dt; }
    
    void set_keyframe(bool is_keyframe) { m_is_keyframe = is_keyframe; }
    
    // Reference keyframe management
    void set_reference_keyframe(std::shared_ptr<Frame> reference_kf);
    std::shared_ptr<Frame> get_reference_keyframe() const;
    const Eigen::Matrix4f& get_relative_transform() const { return m_T_relative_from_ref; }
    void set_relative_transform(const Eigen::Matrix4f& T_relative) { m_T_relative_from_ref = T_relative; }
    
    // Static keyframe management
    static void set_last_keyframe(std::shared_ptr<Frame> keyframe) { m_last_keyframe = keyframe; }
    static std::shared_ptr<Frame> get_last_keyframe() { return m_last_keyframe; }

    // Feature management
    void add_feature(std::shared_ptr<Feature> feature);
    void remove_feature(int feature_id);
    std::shared_ptr<Feature> get_feature(int feature_id);
    std::shared_ptr<const Feature> get_feature(int feature_id) const;
    size_t get_feature_count() const { return m_features.size(); }
    int get_feature_index(int feature_id) const;  // Get index by feature ID
    
    // MapPoint management
    void initialize_map_points(); // Initialize map_points vector with left feature count
    void set_map_point(int feature_index, std::shared_ptr<MapPoint> map_point);
    std::shared_ptr<MapPoint> get_map_point(int feature_index) const;
    bool has_map_point(int feature_index) const;
    const std::vector<std::shared_ptr<MapPoint>>& get_map_points() const { return m_map_points; }
    std::vector<std::shared_ptr<MapPoint>>& get_map_points_mutable() { return m_map_points; }

    // Outlier flag management
    void set_outlier_flag(int feature_index, bool is_outlier);
    bool get_outlier_flag(int feature_index) const;
    const std::vector<bool>& get_outlier_flags() const { return m_outlier_flags; }
    void initialize_outlier_flags(); // Initialize outlier flags with false

    // Camera parameter management
    void set_camera_intrinsics(double fx, double fy, double cx, double cy);
    void get_camera_intrinsics(double& fx, double& fy, double& cx, double& cy) const;
    void set_distortion_coeffs(const std::vector<double>& distortion_coeffs);
    const std::vector<double>& get_distortion_coeffs() const { return m_distortion_coeffs; }
    
    // Camera extrinsics operations
    void set_T_CB(const Eigen::Matrix4d& T_CB) { m_T_CB = T_CB; }
    const Eigen::Matrix4d& get_T_CB() const { return m_T_CB; }
    
    // Undistort a single point
    cv::Point2f undistort_point(const cv::Point2f& distorted_point) const;

    // Feature operations
    void extract_stereo_features(int max_features = 150);
    void compute_stereo_depth();
    
    // Depth operations
    double get_depth(int feature_index) const;
    bool has_depth(int feature_index) const;
    bool has_valid_stereo_depth(const cv::Point2f& pixel_coord) const;
    
    // Visualization functions (for viewer)
    cv::Mat draw_features() const;
    cv::Mat draw_tracks(const Frame& previous_frame) const;
    cv::Mat draw_stereo_matches() const;

    // IMU data management
    void set_imu_data_from_last_frame(const std::vector<IMUData>& imu_data);
    const std::vector<IMUData>& get_imu_data_from_last_frame() const { return m_imu_vec_from_last_frame; }
    std::vector<IMUData>& get_imu_data_from_last_frame_mutable() { return m_imu_vec_from_last_frame; }
    bool has_imu_data() const { return !m_imu_vec_from_last_frame.empty(); }
    size_t get_imu_data_count() const { return m_imu_vec_from_last_frame.size(); }
    
    // Keyframe IMU data management
    void set_imu_data_since_last_keyframe(const std::vector<IMUData>& imu_data);
    const std::vector<IMUData>& get_imu_data_since_last_keyframe() const { return m_imu_vec_since_last_keyframe; }
    bool has_keyframe_imu_data() const { return !m_imu_vec_since_last_keyframe.empty(); }
    size_t get_keyframe_imu_data_count() const { return m_imu_vec_since_last_keyframe.size(); }
    
    // IMU preintegration management
    void set_imu_preintegration_from_last_keyframe(std::shared_ptr<IMUPreintegration> preintegration);
    std::shared_ptr<IMUPreintegration> get_imu_preintegration_from_last_keyframe() const { return m_imu_preintegration_from_last_keyframe; }
    bool has_imu_preintegration_from_last_keyframe() const { return m_imu_preintegration_from_last_keyframe != nullptr; }
    
    void set_imu_preintegration_from_last_frame(std::shared_ptr<IMUPreintegration> preintegration);
    std::shared_ptr<IMUPreintegration> get_imu_preintegration_from_last_frame() const { return m_imu_preintegration_from_last_frame; }
    bool has_imu_preintegration_from_last_frame() const { return m_imu_preintegration_from_last_frame != nullptr; }

private:
    // Frame information
    long long m_timestamp;         // Timestamp in nanoseconds
    int m_frame_id;               // Unique frame ID
    cv::Mat m_left_image;          // Left camera grayscale image
    cv::Mat m_right_image;         // Right camera grayscale image (always provided)
    
    // Features
    std::vector<std::shared_ptr<Feature>> m_features;      // Left camera features
    std::unordered_map<int, size_t> m_feature_id_to_index;  // Quick lookup
    
    // Stereo matches (left feature index -> right feature index)
    std::vector<int> m_stereo_matches; // -1 if no match
    std::vector<double> m_depths;      // Depth for each left feature
    
    // MapPoints corresponding to features (same indexing as m_features)
    std::vector<std::shared_ptr<MapPoint>> m_map_points;
    
    // Outlier flags for map points (same indexing as m_features and m_map_points)
    std::vector<bool> m_outlier_flags;

    // Camera intrinsic parameters
    double m_fx, m_fy;           // Focal lengths
    double m_cx, m_cy;           // Principal point
    std::vector<double> m_distortion_coeffs; // Distortion coefficients [k1, k2, p1, p2, k3]
    
    // Camera extrinsics (body to camera transformation)
    Eigen::Matrix4d m_T_CB;      // Transform from camera to body frame (T_CB = T_BC.inverse())

    // Pose (camera pose in world frame)
    Eigen::Matrix3f m_rotation;    // Rotation matrix (DEPRECATED - use reference keyframe approach)
    Eigen::Vector3f m_translation; // Translation vector (DEPRECATED - use reference keyframe approach)
    bool m_is_keyframe;           // Whether this is a keyframe
    
    // VIO-specific members for optimization
    Sophus::SE3f m_world_pose;     // SE3 world pose for optimization
    Eigen::Vector3f m_velocity;    // Velocity in world frame
    Eigen::Vector3f m_accel_bias;  // Accelerometer bias
    Eigen::Vector3f m_gyro_bias;   // Gyroscope bias
    
    // IMU time management
    double m_dt_from_last_keyframe; // Time difference from last keyframe to this keyframe (seconds)
    
    // Reference keyframe approach for pose management
    std::weak_ptr<Frame> m_reference_keyframe;  // Reference keyframe (weak_ptr to avoid cycles)
    Eigen::Matrix4f m_T_relative_from_ref;      // Transform from reference keyframe to this frame
    static std::shared_ptr<Frame> m_last_keyframe; // Last created keyframe (shared across all frames)

    // Thread safety
    mutable std::mutex m_pose_mutex; // Mutex for pose operations

    // Feature detection parameters
    double m_quality_level = 0.01;
    double m_min_distance = 30.0;

    // IMU data from last frame to current frame
    std::vector<IMUData> m_imu_vec_from_last_frame;
    
    // IMU data accumulated since last keyframe
    std::vector<IMUData> m_imu_vec_since_last_keyframe;
    
    // Preintegrated IMU measurements from last keyframe to current frame
    std::shared_ptr<IMUPreintegration> m_imu_preintegration_from_last_keyframe;
    
    // Preintegrated IMU measurements from last frame to current frame
    std::shared_ptr<IMUPreintegration> m_imu_preintegration_from_last_frame;

    // Helper functions
    void update_feature_index();
    bool is_in_border(const cv::Point2f& point, int border_size = 1) const;
    
    // Internal processing methods
    void extract_features(int max_features = 150);
    void compute_stereo_matches();
    void undistort_features();
    void triangulate_stereo_points();
    double compute_disparity_at_point(const cv::Point2f& pixel_coord) const;
    
    // Visualization function (internal use only)
    cv::Mat draw_rectified_stereo_matches() const;
};

} // namespace lightweight_vio
