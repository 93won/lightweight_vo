#pragma once

#include <memory>
#include <vector>
#include <optional>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "Optimizer.h"

// Forward declarations
namespace lightweight_vio {
    class Frame;
    class MapPoint;
    class FeatureTracker;
}

namespace lightweight_vio {

/**
 * @brief Result of VIO estimation for a single frame
 */
struct EstimationResult {
    bool success = false;
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    int num_features = 0;
    int num_inliers = 0;
    int num_outliers = 0;
    bool is_keyframe = false;
    double optimization_time_ms = 0.0;
    
    // Optional: Additional information
    // OptimizationResult pose_optimization_result; // Commented out - not used in current implementation
};

/**
 * @brief Main VIO estimator class that handles the complete pipeline
 */
class Estimator {
public:
    /**
     * @brief VIO estimation result
     */
    struct EstimationResult {
        bool success;
        Eigen::Matrix4f pose;
        int num_features;
        int num_inliers;
        int num_outliers;
        int num_new_map_points;
        int num_tracked_features;
        int num_features_with_map_points;
        double optimization_time_ms;
        
        EstimationResult() : success(false), pose(Eigen::Matrix4f::Identity()),
                           num_features(0), num_inliers(0), num_outliers(0),
                           num_new_map_points(0), num_tracked_features(0), 
                           num_features_with_map_points(0), optimization_time_ms(0.0) {}
    };

    /**
     * @brief Constructor
     */
    Estimator();
    
    /**
     * @brief Destructor
     */
    ~Estimator() = default;

    /**
     * @brief Process a new stereo frame
     * @param left_image Left stereo image
     * @param right_image Right stereo image  
     * @param timestamp Frame timestamp in nanoseconds
     * @return Estimation result
     */
    EstimationResult process_frame(const cv::Mat& left_image, const cv::Mat& right_image, long long timestamp);

    /**
     * @brief Reset the estimator state
     */
    void reset();

    /**
     * @brief Get current pose
     * @return Current camera pose (Twb)
     */
    Eigen::Matrix4f get_current_pose() const;

    /**
     * @brief Set initial ground truth pose for first frame
     * @param gt_pose Ground truth pose (Twb)
     */
    void set_initial_gt_pose(const Eigen::Matrix4f& gt_pose);
    
    /**
     * @brief Apply ground truth pose to current frame (for debugging)
     * @param gt_pose Ground truth pose (Twb)
     */
    void apply_gt_pose_to_current_frame(const Eigen::Matrix4f& gt_pose);

    /**
     * @brief Get all keyframes
     * @return Vector of keyframes
     */
    const std::vector<std::shared_ptr<Frame>>& get_keyframes() const { return m_keyframes; }

    /**
     * @brief Get all processed frames (for trajectory export)
     * @return Vector of all processed frames
     */
    const std::vector<std::shared_ptr<Frame>>& get_all_frames() const { return m_all_frames; }

    /**
     * @brief Get all map points
     * @return Vector of map points
     */
    const std::vector<std::shared_ptr<MapPoint>>& get_map_points() const { return m_map_points; }

    /**
     * @brief Get current frame
     * @return Current frame (can be nullptr)
     */
    std::shared_ptr<Frame> get_current_frame() const { return m_current_frame; }

private:
    // System components
    std::unique_ptr<FeatureTracker> m_feature_tracker;
    std::unique_ptr<PnPOptimizer> m_pose_optimizer;
    std::unique_ptr<SlidingWindowOptimizer> m_sliding_window_optimizer;
    
    // State
    std::shared_ptr<Frame> m_current_frame;
    std::shared_ptr<Frame> m_previous_frame;
    std::vector<std::shared_ptr<Frame>> m_keyframes;
    std::vector<std::shared_ptr<Frame>> m_all_frames;  // All processed frames for trajectory export
    std::vector<std::shared_ptr<MapPoint>> m_map_points;
    
    // Frame management
    int m_frame_id_counter;
    int m_frames_since_last_keyframe;
    
    // Keyframe management state
    double m_last_keyframe_grid_coverage = 0.0;  // Grid coverage of the last keyframe
    
    // Current pose
    Eigen::Matrix4f m_current_pose;
    
    // Constant velocity model for pose prediction
    Eigen::Matrix4f m_transform_from_last;  // Transformation from last frame to current frame
    
    // Ground truth initialization
    bool m_has_initial_gt_pose;
    Eigen::Matrix4f m_initial_gt_pose;
    
    /**
     * @brief Initialize a new stereo frame
     * @param left_image Left stereo image
     * @param right_image Right stereo image
     * @param timestamp Frame timestamp
     * @return New frame
     */
    std::shared_ptr<Frame> create_frame(const cv::Mat& left_image, const cv::Mat& right_image, long long timestamp);
    
    /**
     * @brief Predict current frame pose using constant velocity model
     */
    void predict_state();
    
    /**
     * @brief Update transform from last frame for velocity estimation
     */
    void update_transform_from_last();
    
    /**
     * @brief Create initial map points from stereo or motion
     * @param frame Frame to create map points for
     * @return Number of created map points
     */
    int create_initial_map_points(std::shared_ptr<Frame> frame);
    
    /**
     * @brief Create new map points for untracked stereo features
     * @param frame Frame to create map points for
     * @return Number of created map points
     */
    int create_new_map_points(std::shared_ptr<Frame> frame);
    
    /**
     * @brief Count how many features have associated map points
     * @param frame Frame to count
     * @return Number of features with map points
     */
    int count_features_with_map_points(std::shared_ptr<Frame> frame);
    
    /**
     * @brief Decide whether to create a new keyframe
     * @param frame Current frame
     * @return True if should create keyframe
     */
    bool should_create_keyframe(std::shared_ptr<Frame> frame);
    
    /**
     * @brief Calculate grid coverage ratio with features that have map points
     * @param frame Frame to analyze
     * @return Ratio of grid cells that have features with map points (0.0 to 1.0)
     */
    double calculate_grid_coverage_with_map_points(std::shared_ptr<Frame> frame);
    
    
    /**
     * @brief Create a new keyframe
     * @param frame Frame to convert to keyframe
     */
    void create_keyframe(std::shared_ptr<Frame> frame);
    
    /**
     * @brief Perform pose optimization
     * @param frame Frame to optimize pose for
     * @return Optimization result
     */
    OptimizationResult optimize_pose(std::shared_ptr<Frame> frame);
    
    /**
     * @brief Compute reprojection error statistics for map points
     * @param frame Frame to compute reprojection errors for
     */
    void compute_reprojection_error_statistics(std::shared_ptr<Frame> frame);
};

} // namespace lightweight_vio
