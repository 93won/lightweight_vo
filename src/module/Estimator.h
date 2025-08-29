#pragma once

#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <module/PoseOptimizer.h>

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
    OptimizationResult pose_optimization_result;
};

/**
 * @brief Main VIO estimator class that handles the complete pipeline
 */
class Estimator {
public:
    /**
     * @brief Configuration for the VIO estimator
     */
    struct Config {
        // Feature tracking parameters
        int max_features = 150;
        double quality_level = 0.01;
        double min_distance = 30.0;
        
        // Camera parameters (EuRoC dataset)
        double fx = 458.654;
        double fy = 457.296;
        double cx = 367.215;
        double cy = 248.375;
        double baseline = 0.11; // Stereo baseline in meters
        std::vector<double> distortion_coeffs = {-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05};
        
        // Pose optimization parameters
        PoseOptimizer::Config pose_optimizer_config;
        
        // System parameters
        bool enable_pose_optimization = true;
        bool enable_visualization = true;
        int keyframe_interval = 10;  // Create keyframe every N frames
        
        Config() {
            // Default pose optimizer configuration
            pose_optimizer_config.enable_outlier_detection = true;
            pose_optimizer_config.outlier_detection_rounds = 3;
            pose_optimizer_config.print_summary = false;
        }
    };

    /**
     * @brief VIO estimation result
     */
    struct EstimationResult {
        bool success;
        Eigen::Matrix4f pose;
        int num_features;
        int num_inliers;
        int num_outliers;
        double optimization_time_ms;
        
        EstimationResult() : success(false), pose(Eigen::Matrix4f::Identity()),
                           num_features(0), num_inliers(0), num_outliers(0),
                           optimization_time_ms(0.0) {}
    };

    /**
     * @brief Constructor
     * @param config Estimator configuration
     */
    Estimator(const Config& config = Config());
    
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
     * @brief Get all keyframes
     * @return Vector of keyframes
     */
    const std::vector<std::shared_ptr<Frame>>& get_keyframes() const { return m_keyframes; }

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

    /**
     * @brief Set configuration
     * @param config New configuration
     */
    void set_config(const Config& config) { m_config = config; }

    /**
     * @brief Get configuration
     * @return Current configuration
     */
    const Config& get_config() const { return m_config; }

private:
    // Configuration
    Config m_config;
    
    // System components
    std::unique_ptr<FeatureTracker> m_feature_tracker;
    std::unique_ptr<PoseOptimizer> m_pose_optimizer;
    
    // State
    std::shared_ptr<Frame> m_current_frame;
    std::shared_ptr<Frame> m_previous_frame;
    std::vector<std::shared_ptr<Frame>> m_keyframes;
    std::vector<std::shared_ptr<MapPoint>> m_map_points;
    
    // Frame management
    int m_frame_id_counter;
    int m_frames_since_last_keyframe;
    
    // Current pose
    Eigen::Matrix4f m_current_pose;
    
    /**
     * @brief Initialize a new stereo frame
     * @param left_image Left stereo image
     * @param right_image Right stereo image
     * @param timestamp Frame timestamp
     * @return New frame
     */
    std::shared_ptr<Frame> create_frame(const cv::Mat& left_image, const cv::Mat& right_image, long long timestamp);
    
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
     * @brief Associate tracked features with existing map points from previous frames
     * @param frame Frame to associate features
     * @return Number of associated features
     */
    int associate_tracked_features_with_map_points(std::shared_ptr<Frame> frame);
    
    /**
     * @brief Associate features with existing map points
     * @param frame Frame to associate features
     * @return Number of associated features
     */
    int associate_features_with_map_points(std::shared_ptr<Frame> frame);
    
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
     * @brief Update map points based on optimization results
     * @param frame Frame with updated pose
     */
    void update_map_points(std::shared_ptr<Frame> frame);
};

} // namespace lightweight_vio
