/**
 * @file      Feature.cpp
 * @brief     Implements the Feature class.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-08-11
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "database/Feature.h"
#include <cmath>

namespace lightweight_vio {

Feature::Feature(int feature_id, const cv::Point2f& pixel_coord)
    : m_feature_id(feature_id)
    , m_tracked_feature_id(-1)  // No tracked feature by default
    , m_pixel_coord(pixel_coord)
    , m_undistorted_coord(cv::Point2f(-1, -1))  // Invalid until computed
    , m_normalized_coord(Eigen::Vector2f::Zero())
    , m_3d_point(Eigen::Vector3f::Zero())
    , m_velocity(Eigen::Vector2f::Zero())
    , m_track_count(1)
    , m_depth(-1.0f)  // Invalid depth initially
    , m_is_valid(true)
    , m_has_3d_point(false)
    , m_right_coord(cv::Point2f(-1, -1))  // Invalid stereo coordinate initially
    , m_right_undistorted_coord(cv::Point2f(-1, -1))  // Invalid undistorted stereo coordinate initially
    , m_disparity(-1.0f)  // Invalid disparity initially
    , m_undistorted_disparity(-1.0f)  // Invalid undistorted disparity initially
    , m_has_stereo_match(false)
{
}

float Feature::calculate_parallax(const Feature& other) const {
    Eigen::Vector2f diff = m_normalized_coord - other.m_normalized_coord;
    return diff.norm();
}

} // namespace lightweight_vio
