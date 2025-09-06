/**
 * @file      MapPoint.h
 * @brief     Defines the MapPoint class, representing a 3D point in the map.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-08-18
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <mutex>

namespace lightweight_vio {

class Frame;
class Feature;

struct Observation {
    std::weak_ptr<Frame> frame;
    int feature_index;
    
    Observation(std::shared_ptr<Frame> f, int idx) 
        : frame(f), feature_index(idx) {}
};

class MapPoint {
public:
    MapPoint();
    MapPoint(const Eigen::Vector3f& position);
    ~MapPoint();

    // Position management
    void set_position(const Eigen::Vector3f& position);
    const Eigen::Vector3f& get_position() const;
    
    // Observation management
    void add_observation(std::shared_ptr<Frame> frame, int feature_index);
    void remove_observation(std::shared_ptr<Frame> frame);
    const std::vector<Observation>& get_observations() const;
    int get_observation_count() const;
    
    // Utility functions
    bool is_observed_by_frame(std::shared_ptr<Frame> frame) const;
    int get_feature_index_in_frame(std::shared_ptr<Frame> frame) const;
    
    // MapPoint management
    void set_id(int id);
    int get_id() const;
    
    void set_bad();
    bool is_bad() const;
    
    // Multi-view triangulation flag
    void set_multi_view_triangulated(bool flag);
    bool is_multi_view_triangulated() const;
    
    // Triangulation and refinement
    void update_position_from_observations();
    double compute_reprojection_error() const;

private:
    int m_id;
    Eigen::Vector3f m_position;
    std::vector<Observation> m_observations;
    bool m_is_bad;
    bool m_is_multi_view_triangulated;
    
    // Thread safety
    mutable std::mutex m_position_mutex; // Mutex for position operations
    mutable std::mutex m_data_mutex;     // Mutex for other data operations
    
    static int s_next_id;
};

} // namespace lightweight_vio
