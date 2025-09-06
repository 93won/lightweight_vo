#pragma once

#include <memory>
#include <vector>
#include <mutex>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <factor/Parameters.h>
#include <factor/Factors.h>

// Forward declarations
namespace lightweight_vio {
    class Frame;
    class MapPoint;
    
    namespace factor {
        class PnPFactor;
        struct CameraParameters;
    }
}

namespace lightweight_vio {

/**
 * @brief Pose optimization result
 */
struct OptimizationResult {
    bool success;
    int num_inliers;
    int num_outliers;
    Eigen::Matrix4f optimized_pose;
    std::vector<bool> outlier_flags;
    double initial_cost;
    double final_cost;
    int num_iterations;
    
    OptimizationResult() : success(false), num_inliers(0), num_outliers(0), 
                          initial_cost(0.0), final_cost(0.0), num_iterations(0) {}
};

/**
 * @brief Structure to hold both residual block ID and cost function
 */
struct ObservationInfo {
    ceres::ResidualBlockId residual_id;
    factor::PnPFactor* cost_function;
    
    ObservationInfo(ceres::ResidualBlockId id, factor::PnPFactor* func)
        : residual_id(id), cost_function(func) {}
};

/**
 * @brief Pose optimizer for VIO system using Ceres
 */
class PnPOptimizer {
public:
    /**
     * @brief Constructor
     */
    PnPOptimizer();
    
    /**
     * @brief Destructor
     */
    ~PnPOptimizer() = default;
    
    /**
     * @brief Optimize pose using PnP with map points
     * @param frame Frame to optimize pose for
     * @return Optimization result
     */
    OptimizationResult optimize_pose(std::shared_ptr<Frame> frame);

private:
    /**
     * @brief Add monocular PnP observation to problem
     * @param problem Ceres problem
     * @param pose_params Pose parameters
     * @param world_point 3D world point
     * @param observation 2D observation
     * @param camera_params Camera parameters
     * @param frame Frame containing the T_CB transformation
     * @return Observation info with residual block ID and cost function
     */
    ObservationInfo add_mono_observation(
        ceres::Problem& problem,
        double* pose_params,
        const Eigen::Vector3d& world_point,
        const Eigen::Vector2d& observation,
        const factor::CameraParameters& camera_params,
        std::shared_ptr<Frame> frame,
        double pixel_noise_std = 1.0);
    
    /**
     * @brief Add a mono observation residual with observation-based weighting
     * @param problem Ceres problem instance
     * @param pose_params Pose parameters (6-DOF SE3)
     * @param world_point 3D world point
     * @param observation 2D image observation
     * @param camera_params Camera intrinsics
     * @param frame Frame containing the observation
     * @param pixel_noise_std Standard deviation of pixel noise
     * @param num_observations Number of observations for weighting
     * @return Observation info with residual block ID and cost function
     */
    ObservationInfo add_mono_observation(
        ceres::Problem& problem,
        double* pose_params,
        const Eigen::Vector3d& world_point,
        const Eigen::Vector2d& observation,
        const factor::CameraParameters& camera_params,
        std::shared_ptr<Frame> frame,
        double pixel_noise_std,
        int num_observations);
    
    /**
     * @brief Detect outliers using chi-square test
     * @param pose_params Current pose parameters
     * @param observations Vector of observation info
     * @param feature_indices Indices of features corresponding to observations
     * @param frame Frame to update outlier flags
     * @return Number of inliers
     */
    int detect_outliers(double const* const* pose_params,
                       const std::vector<ObservationInfo>& observations,
                       const std::vector<int>& feature_indices,
                       std::shared_ptr<Frame> frame);
    
    /**
     * @brief Setup solver options based on configuration
     * @return Configured solver options
     */
    ceres::Solver::Options setup_solver_options() const;
    
    /**
     * @brief Convert frame pose to SE3 tangent space
     * @param frame Frame with pose to convert
     * @return SE3 tangent space vector [6]
     */
    Eigen::Vector6d frame_to_se3_tangent(std::shared_ptr<Frame> frame) const;
    
    /**
     * @brief Convert SE3 tangent space to 4x4 matrix
     * @param se3_tangent SE3 tangent space vector [6]
     * @return 4x4 transformation matrix
     */
    Eigen::Matrix4f se3_tangent_to_matrix(const Eigen::Vector6d& se3_tangent) const;
    
    /**
     * @brief Create robust loss function
     * @param delta Huber loss delta parameter
     * @return Robust loss function
     */
    ceres::LossFunction* create_robust_loss(double delta) const;
    
    /**
     * @brief Create information matrix for pixel observations
     * @param pixel_noise Standard deviation of pixel noise
     * @return 2x2 information matrix
     */
    Eigen::Matrix2d create_information_matrix(double pixel_noise = 1.0) const;
    
    /**
     * @brief Create information matrix with observation-based weighting
     * @param pixel_noise Standard deviation of pixel noise
     * @param num_observations Number of observations for the map point (max 3.0)
     * @return 2x2 information matrix
     */
    Eigen::Matrix2d create_information_matrix(double pixel_noise, int num_observations) const;

private:
    // Global mutexes for thread-safe access to MapPoints and Frames
    static std::mutex s_mappoint_mutex;
    static std::mutex s_keyframe_mutex;
};

/**
 * @brief Sliding Window Bundle Adjustment result
 */
struct SlidingWindowResult {
    bool success;
    int num_inliers;
    int num_outliers;
    int num_poses_optimized;
    int num_points_optimized;
    double initial_cost;
    double final_cost;
    int num_iterations;
    
    SlidingWindowResult() : success(false), num_inliers(0), num_outliers(0),
                           num_poses_optimized(0), num_points_optimized(0),
                           initial_cost(0.0), final_cost(0.0), num_iterations(0) {}
};

/**
 * @brief Structure to hold bundle adjustment observation info
 */
struct BAObservationInfo {
    ceres::ResidualBlockId residual_id;
    factor::BAFactor* cost_function;
    int keyframe_index;  // Index in sliding window
    int mappoint_index;  // Index in map points vector
    
    BAObservationInfo(ceres::ResidualBlockId id, factor::BAFactor* func, 
                     int kf_idx, int mp_idx)
        : residual_id(id), cost_function(func), 
          keyframe_index(kf_idx), mappoint_index(mp_idx) {}
};

/**
 * @brief Sliding Window Bundle Optimizer for VIO system using Ceres
 */
class SlidingWindowOptimizer {
public:
    /**
     * @brief Constructor
     * @param window_size Maximum number of keyframes in sliding window
     */
    SlidingWindowOptimizer(size_t window_size = 10);
    
    /**
     * @brief Destructor
     */
    ~SlidingWindowOptimizer() = default;
    
    /**
     * @brief Optimize sliding window of keyframes and map points
     * @param keyframes Vector of keyframes in sliding window (oldest to newest)
     * @param force_stop_flag Optional flag to force early termination
     * @return Optimization result
     */
    SlidingWindowResult optimize(const std::vector<std::shared_ptr<Frame>>& keyframes,
                                bool* force_stop_flag = nullptr);

private:
    /**
     * @brief Collect map points observed by keyframes in sliding window
     * @param keyframes Vector of keyframes
     * @return Vector of unique map points
     */
    std::vector<std::shared_ptr<MapPoint>> collect_window_map_points(
        const std::vector<std::shared_ptr<Frame>>& keyframes) const;
    
    /**
     * @brief Add bundle adjustment observation to problem
     * @param problem Ceres problem
     * @param pose_params Pose parameters for keyframe
     * @param point_params 3D point parameters
     * @param observation 2D observation
     * @param camera_params Camera parameters
     * @param frame Frame containing the T_CB transformation
     * @param kf_index Keyframe index in sliding window
     * @param mp_index Map point index
     * @return Observation info with residual block ID and cost function
     */
    BAObservationInfo add_ba_observation(
        ceres::Problem& problem,
        double* pose_params,
        double* point_params,
        const Eigen::Vector2d& observation,
        const factor::CameraParameters& camera_params,
        std::shared_ptr<Frame> frame,
        int kf_index,
        int mp_index,
        double pixel_noise_std = 1.0);
    
    /**
     * @brief Add BA observation with observation-based information weighting
     * @param problem Ceres optimization problem
     * @param pose_params Pose parameters (SE3 tangent space)
     * @param point_params 3D point parameters
     * @param observation 2D pixel observation
     * @param camera_params Camera parameters
     * @param frame Frame containing the T_CB transformation
     * @param kf_index Keyframe index in sliding window
     * @param mp_index Map point index
     * @param pixel_noise_std Standard deviation of pixel noise
     * @param num_observations Number of times this map point has been observed
     * @return Observation info with residual block ID and cost function
     */
    BAObservationInfo add_ba_observation(
        ceres::Problem& problem,
        double* pose_params,
        double* point_params,
        const Eigen::Vector2d& observation,
        const factor::CameraParameters& camera_params,
        std::shared_ptr<Frame> frame,
        int kf_index,
        int mp_index,
        double pixel_noise_std,
        int num_observations);
    
    /**
     * @brief Setup optimization problem with keyframes and map points
     * @param problem Ceres problem
     * @param keyframes Vector of keyframes
     * @param map_points Vector of map points
     * @param pose_params_vec Vector of pose parameter arrays
     * @param point_params_vec Vector of point parameter arrays
     * @return Vector of observation info
     */
    std::vector<BAObservationInfo> setup_optimization_problem(
        ceres::Problem& problem,
        const std::vector<std::shared_ptr<Frame>>& keyframes,
        const std::vector<std::shared_ptr<MapPoint>>& map_points,
        std::vector<std::vector<double>>& pose_params_vec,
        std::vector<std::vector<double>>& point_params_vec);
    
    /**
     * @brief Apply marginalization strategy
     * @param problem Ceres problem
     * @param keyframes Vector of keyframes
     * @param map_points Vector of map points
     * @param pose_params_vec Vector of pose parameter arrays
     * @param point_params_vec Vector of point parameter arrays
     */
    void apply_marginalization_strategy(
        ceres::Problem& problem,
        const std::vector<std::shared_ptr<Frame>>& keyframes,
        const std::vector<std::shared_ptr<MapPoint>>& map_points,
        const std::vector<std::vector<double>>& pose_params_vec,
        const std::vector<std::vector<double>>& point_params_vec);
    
    /**
     * @brief Detect outliers using chi-square test for bundle adjustment
     * @param pose_params_vec Vector of pose parameter arrays
     * @param point_params_vec Vector of point parameter arrays
     * @param observations Vector of observation info
     * @param keyframes Vector of keyframes to update outlier flags
     * @return Number of inliers
     */
    int detect_ba_outliers(
        const std::vector<std::vector<double>>& pose_params_vec,
        const std::vector<std::vector<double>>& point_params_vec,
        const std::vector<BAObservationInfo>& observations,
        const std::vector<std::shared_ptr<Frame>>& keyframes,
        const std::vector<std::shared_ptr<MapPoint>>& map_points);
    
    /**
     * @brief Update keyframes and map points with optimized values
     * @param keyframes Vector of keyframes to update
     * @param map_points Vector of map points to update
     * @param pose_params_vec Vector of optimized pose parameters
     * @param point_params_vec Vector of optimized point parameters
     */
    void update_optimized_values(
        const std::vector<std::shared_ptr<Frame>>& keyframes,
        const std::vector<std::shared_ptr<MapPoint>>& map_points,
        const std::vector<std::vector<double>>& pose_params_vec,
        const std::vector<std::vector<double>>& point_params_vec);
    
    /**
     * @brief Setup solver options for sliding window optimization
     * @return Configured solver options
     */
    ceres::Solver::Options setup_solver_options() const;
    
    /**
     * @brief Create robust loss function
     * @param delta Huber loss delta parameter
     * @return Robust loss function
     */
    ceres::LossFunction* create_robust_loss(double delta) const;
    
    /**
     * @brief Create information matrix for pixel observations
     * @param pixel_noise Standard deviation of pixel noise
     * @return 2x2 information matrix
     */
    Eigen::Matrix2d create_information_matrix(double pixel_noise = 1.0) const;
    
    /**
     * @brief Create information matrix with observation-based weighting
     * @param pixel_noise Standard deviation of pixel noise
     * @param num_observations Number of observations for weighting (max 3.0)
     * @return 2x2 information matrix
     */
    Eigen::Matrix2d create_information_matrix(double pixel_noise, int num_observations) const;

private:
    size_t m_window_size;        // Maximum keyframes in sliding window
    int m_max_iterations;        // Maximum optimization iterations
    double m_huber_delta;        // Huber loss delta parameter
    double m_pixel_noise_std;    // Pixel noise standard deviation
    double m_outlier_threshold;  // Chi-square outlier threshold
    
    // Global mutexes for thread-safe access to MapPoints and Frames
    static std::mutex s_mappoint_mutex;
    static std::mutex s_keyframe_mutex;
};

} // namespace lightweight_vio
