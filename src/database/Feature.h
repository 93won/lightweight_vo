#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>

namespace lightweight_vio {

class Feature {
public:
    Feature(int feature_id, const cv::Point2f& pixel_coord);
    ~Feature() = default;

        // Getters
    int get_feature_id() const { return m_feature_id; }
    cv::Point2f get_pixel_coord() const { return m_pixel_coord; }
    Eigen::Vector2f get_normalized_coord() const { return m_normalized_coord; }
    Eigen::Vector2f get_velocity() const { return m_velocity; }
    float get_depth() const { return m_depth; }
    int get_track_count() const { return m_track_count; }
    bool is_valid() const { return m_is_valid; }

    // Setters
    void set_pixel_coord(const cv::Point2f& coord) { m_pixel_coord = coord; }
    void set_normalized_coord(const Eigen::Vector2f& coord) { m_normalized_coord = coord; }
    void set_velocity(const Eigen::Vector2f& velocity) { m_velocity = velocity; }
    void set_depth(float depth) { m_depth = depth; }
    void set_track_count(int count) { m_track_count = count; }
    void set_valid(bool valid) { m_is_valid = valid; }

    // Operations
    void increment_track_count() { m_track_count++; }
    void reset_track_count() { m_track_count = 1; }
    
    // Stereo operations
    void set_stereo_match(const cv::Point2f& right_coord, float disparity) {
        m_right_coord = right_coord;
        m_disparity = disparity;
        m_has_stereo_match = true;
    }
    bool has_stereo_match() const { return m_has_stereo_match; }
    const cv::Point2f& get_right_coord() const { return m_right_coord; }
    float get_stereo_disparity() const { return m_disparity; }
    
    // Calculate parallax between two observations
    float calculate_parallax(const Feature& other) const;

private:
    int m_feature_id;              // Unique feature ID
    cv::Point2f m_pixel_coord;      // Pixel coordinates in left image
    Eigen::Vector2f m_normalized_coord;  // Normalized camera coordinates
    Eigen::Vector2f m_velocity;    // Optical flow velocity
    int m_track_count;             // Number of times tracked
    float m_depth;                 // Estimated depth (inverse depth parameterization)
    bool m_is_valid;               // Whether this feature is valid
    
    // Stereo matching data
    cv::Point2f m_right_coord;     // Pixel coordinates in right image
    float m_disparity;             // Stereo disparity
    bool m_has_stereo_match;
};

} // namespace lightweight_vio
