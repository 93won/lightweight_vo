#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include "../database/Frame.h"
#include "../database/Feature.h"
#include "../util/Config.h"

namespace lightweight_vio {

class FeatureTracker {
public:
    FeatureTracker();
    ~FeatureTracker() = default;

    // Main tracking function
    void track_features(std::shared_ptr<Frame> current_frame, 
                       std::shared_ptr<Frame> previous_frame = nullptr);

    // Feature extraction and tracking
    void extract_new_features(std::shared_ptr<Frame> frame);
    void optical_flow_tracking(std::shared_ptr<Frame> current_frame, 
                              std::shared_ptr<Frame> previous_frame);
    
    // Outlier rejection
    void reject_outliers_with_fundamental_matrix(std::shared_ptr<Frame> current_frame,
                                               std::shared_ptr<Frame> previous_frame);

    // Feature distribution
    void set_mask(std::shared_ptr<Frame> frame);

private:
    // Configuration reference
    const Config& m_config = Config::getInstance();
    
    // Global feature ID counter
    int m_global_feature_id;
    
    // Helper functions
    bool is_in_border(const cv::Point2f& point, const cv::Size& img_size, int border_size = 1) const;
    void update_feature_track_count(std::shared_ptr<Frame> frame);
    std::vector<cv::Point2f> extract_points_from_features(const std::vector<std::shared_ptr<Feature>>& features);
    void update_features_with_points(std::vector<std::shared_ptr<Feature>>& features, 
                                    const std::vector<cv::Point2f>& points,
                                    const std::vector<uchar>& status);
};

} // namespace lightweight_vio
