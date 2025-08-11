#include "Feature.h"
#include <cmath>

namespace lightweight_vio {

Feature::Feature(int feature_id, const cv::Point2f& pixel_coord)
    : m_feature_id(feature_id)
    , m_pixel_coord(pixel_coord)
    , m_normalized_coord(Eigen::Vector2f::Zero())
    , m_velocity(Eigen::Vector2f::Zero())
    , m_track_count(1)
    , m_depth(-1.0f)  // Invalid depth initially
    , m_is_valid(true)
    , m_right_coord(cv::Point2f(-1, -1))  // Invalid stereo coordinate initially
    , m_disparity(-1.0f)  // Invalid disparity initially
    , m_has_stereo_match(false)
{
}

float Feature::calculate_parallax(const Feature& other) const {
    Eigen::Vector2f diff = m_normalized_coord - other.m_normalized_coord;
    return diff.norm();
}

} // namespace lightweight_vio
