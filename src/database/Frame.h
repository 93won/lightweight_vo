#pragma once

#include "Feature.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <memory>
#include <chrono>
#include <unordered_map>

namespace lightweight_vio {

class MapPoint;

class Frame {
public:
    Frame(long long timestamp, int frame_id);
    Frame(long long timestamp, int frame_id, double fx, double fy, double cx, double cy, 
          const std::vector<double>& distortion_coeffs = {0.0, 0.0, 0.0, 0.0, 0.0});
    ~Frame() = default;

    // Getters
    long long get_timestamp() const { return m_timestamp; }
    int get_frame_id() const { return m_frame_id; }
    const cv::Mat& get_left_image() const { return m_left_image; }
    const cv::Mat& get_right_image() const { return m_right_image; }
    const cv::Mat& get_image() const { return m_left_image; } // For backward compatibility
    const std::vector<std::shared_ptr<Feature>>& get_features() const { return m_features; }
    const Eigen::Matrix3f& get_rotation() const { return m_rotation; }
    const Eigen::Vector3f& get_translation() const { return m_translation; }
    bool is_keyframe() const { return m_is_keyframe; }
    bool is_stereo() const { return !m_right_image.empty(); }

    // Setters
    void set_left_image(const cv::Mat& image) { m_left_image = image.clone(); }
    void set_right_image(const cv::Mat& image) { m_right_image = image.clone(); }
    void set_image(const cv::Mat& image) { m_left_image = image.clone(); } // For backward compatibility
    void set_stereo_images(const cv::Mat& left_image, const cv::Mat& right_image) {
        m_left_image = left_image.clone();
        m_right_image = right_image.clone();
    }
    void set_pose(const Eigen::Matrix3f& rotation, const Eigen::Vector3f& translation);
    void set_keyframe(bool is_keyframe) { m_is_keyframe = is_keyframe; }

    // Feature management
    void add_feature(std::shared_ptr<Feature> feature);
    void remove_feature(int feature_id);
    std::shared_ptr<Feature> get_feature(int feature_id);
    std::shared_ptr<const Feature> get_feature(int feature_id) const;
    size_t get_feature_count() const { return m_features.size(); }
    
    // MapPoint management
    void initialize_map_points(); // Initialize map_points vector with left feature count
    void set_map_point(int feature_index, std::shared_ptr<MapPoint> map_point);
    std::shared_ptr<MapPoint> get_map_point(int feature_index) const;
    bool has_map_point(int feature_index) const;
    const std::vector<std::shared_ptr<MapPoint>>& get_map_points() const { return m_map_points; }

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
    
    // Undistort a single point
    cv::Point2f undistort_point(const cv::Point2f& distorted_point) const;

    // Feature operations
    void extract_features(int max_features = 150);
    
    // Stereo operations
    void compute_stereo_matches();
    void undistort_features();
    void triangulate_stereo_points();
    
    // Visualization
    cv::Mat draw_features() const;
    cv::Mat draw_tracks(const Frame& previous_frame) const;
    cv::Mat draw_stereo_matches() const;
    cv::Mat draw_rectified_stereo_matches() const;

private:
    // Frame information
    long long m_timestamp;         // Timestamp in nanoseconds
    int m_frame_id;               // Unique frame ID
    cv::Mat m_left_image;          // Left camera grayscale image
    cv::Mat m_right_image;         // Right camera grayscale image (optional for stereo)
    
    // Features
    std::vector<std::shared_ptr<Feature>> m_features;
    std::unordered_map<int, size_t> m_feature_id_to_index;  // Quick lookup
    
    // MapPoints corresponding to features (same indexing as m_features)
    std::vector<std::shared_ptr<MapPoint>> m_map_points;
    
    // Outlier flags for map points (same indexing as m_features and m_map_points)
    std::vector<bool> m_outlier_flags;

    // Camera intrinsic parameters
    double m_fx, m_fy;           // Focal lengths
    double m_cx, m_cy;           // Principal point
    std::vector<double> m_distortion_coeffs; // Distortion coefficients [k1, k2, p1, p2, k3]

    // Pose (camera pose in world frame)
    Eigen::Matrix3f m_rotation;    // Rotation matrix
    Eigen::Vector3f m_translation; // Translation vector
    bool m_is_keyframe;           // Whether this is a keyframe

    // Feature detection parameters
    double m_quality_level = 0.01;
    double m_min_distance = 30.0;

    // Helper functions
    void update_feature_index();
    bool is_in_border(const cv::Point2f& point, int border_size = 1) const;
};

} // namespace lightweight_vio
