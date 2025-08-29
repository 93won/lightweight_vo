#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <util/Config.h>

// Forward declarations
namespace lightweight_vio {
    class Frame;
    class Feature;
    class MapPoint;
}

namespace lightweight_vio {

class FeatureTracker {
public:
    FeatureTracker();
    ~FeatureTracker() = default;

    // Main tracking function
    void track_features(std::shared_ptr<Frame> current_frame, 
                       std::shared_ptr<Frame> previous_frame = nullptr);

    // Feature extraction and tracking
    std::pair<int, int> extract_new_features(std::shared_ptr<Frame> frame);
    std::pair<int, int> optical_flow_tracking(std::shared_ptr<Frame> current_frame, 
                              std::shared_ptr<Frame> previous_frame);
    
    // Outlier rejection
    void reject_outliers(std::shared_ptr<Frame> current_frame,
                         std::shared_ptr<Frame> previous_frame);

    // Feature distribution
    void set_mask(std::shared_ptr<Frame> frame);

private:
    // Configuration reference
    const Config& m_config = Config::getInstance();
    
    // Feature distribution mask
    cv::Mat m_mask;
    
    // Helper functions
    bool is_in_border(const cv::Point2f& point, const cv::Size& img_size, int border_size = 1) const;
    void update_feature_track_count(std::shared_ptr<Frame> frame);
    std::vector<cv::Point2f> extract_points_from_features(const std::vector<std::shared_ptr<Feature>>& features);
    void update_features_with_points(std::vector<std::shared_ptr<Feature>>& features, 
                                    const std::vector<cv::Point2f>& points,
                                    const std::vector<uchar>& status);
    
    // Map point creation and triangulation
    bool can_triangulate_feature(std::shared_ptr<Feature> feature, std::shared_ptr<Frame> frame);
    std::shared_ptr<MapPoint> create_map_point_from_stereo(std::shared_ptr<Feature> feature, std::shared_ptr<Frame> frame);
    
    // Batch stereo matching for all features without map points
    int batch_stereo_matching_and_map_point_creation(const std::shared_ptr<Frame>& frame);
};

} // namespace lightweight_vio
