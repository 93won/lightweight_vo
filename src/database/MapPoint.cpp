/**
 * @file      MapPoint.cpp
 * @brief     Implements the MapPoint class.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-08-18
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "database/MapPoint.h"
#include "database/Frame.h"
#include "database/Feature.h"
#include <algorithm>
#include <iostream>

namespace lightweight_vio {

int MapPoint::s_next_id = 0;

MapPoint::MapPoint() 
    : m_id(s_next_id++)
    , m_position(0.0f, 0.0f, 0.0f)
    , m_is_bad(false)
    , m_is_multi_view_triangulated(false)
{
}

MapPoint::MapPoint(const Eigen::Vector3f& position)
    : m_id(s_next_id++)
    , m_position(position)
    , m_is_bad(false)
    , m_is_multi_view_triangulated(false)
{
}

MapPoint::~MapPoint() {
}

void MapPoint::set_position(const Eigen::Vector3f& position) {
    std::lock_guard<std::mutex> lock(m_position_mutex);
    m_position = position;
}

const Eigen::Vector3f& MapPoint::get_position() const {
    std::lock_guard<std::mutex> lock(m_position_mutex);
    return m_position;
}

void MapPoint::add_observation(std::shared_ptr<Frame> frame, int feature_index) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    if (!frame) return;
    
    // Check if this frame is already observing this map point
    for (auto& obs : m_observations) {
        if (auto existing_frame = obs.frame.lock()) {
            if (existing_frame == frame) {
                // Update feature index if frame already exists
                obs.feature_index = feature_index;
                return;
            }
        }
    }
    
    // Add new observation
    m_observations.emplace_back(frame, feature_index);
}

void MapPoint::remove_observation(std::shared_ptr<Frame> frame) {
    if (!frame) return;
    
    m_observations.erase(
        std::remove_if(m_observations.begin(), m_observations.end(),
                      [frame](const Observation& obs) {
                          if (auto obs_frame = obs.frame.lock()) {
                              return obs_frame == frame;
                          }
                          return true; // Remove expired weak_ptr
                      }),
        m_observations.end()
    );
    
    // Mark as bad if no observations left
    if (m_observations.empty()) {
        set_bad();
    }
}

const std::vector<Observation>& MapPoint::get_observations() const {
    return m_observations;
}

int MapPoint::get_observation_count() const {
    return static_cast<int>(m_observations.size());
}

bool MapPoint::is_observed_by_frame(std::shared_ptr<Frame> frame) const {
    if (!frame) return false;
    
    for (const auto& obs : m_observations) {
        if (auto obs_frame = obs.frame.lock()) {
            if (obs_frame == frame) {
                return true;
            }
        }
    }
    return false;
}

int MapPoint::get_feature_index_in_frame(std::shared_ptr<Frame> frame) const {
    if (!frame) return -1;
    
    for (const auto& obs : m_observations) {
        if (auto obs_frame = obs.frame.lock()) {
            if (obs_frame == frame) {
                return obs.feature_index;
            }
        }
    }
    return -1;
}

void MapPoint::set_id(int id) {
    m_id = id;
}

int MapPoint::get_id() const {
    return m_id;
}

void MapPoint::set_bad() {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_is_bad = true;
}

bool MapPoint::is_bad() const {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    return m_is_bad;
}

void MapPoint::set_multi_view_triangulated(bool flag) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_is_multi_view_triangulated = flag;
}

bool MapPoint::is_multi_view_triangulated() const {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    return m_is_multi_view_triangulated;
}

double MapPoint::compute_reprojection_error() const {
    if (m_observations.empty()) {
        return 0.0;
    }
    
    double total_error = 0.0;
    int valid_observations = 0;
    
    for (const auto& obs : m_observations) {
        if (auto frame = obs.frame.lock()) {
            // TODO: Implement reprojection error calculation
            // This would require camera intrinsics and pose information
            valid_observations++;
        }
    }
    
    return valid_observations > 0 ? total_error / valid_observations : 0.0;
}

} // namespace lightweight_vio
