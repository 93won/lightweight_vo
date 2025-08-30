#pragma once

#include <memory>
#include <vector>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <factor/Parameters.h>
#include <factor/PnPFactors.h>

// Forward declarations
namespace lightweight_vio {
    class Frame;
    class MapPoint;
    
    namespace factor {
        class MonoPnPFactor;
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
    double final_cost;
    int num_iterations;
    
    OptimizationResult() : success(false), num_inliers(0), num_outliers(0), 
                          final_cost(0.0), num_iterations(0) {}
};

/**
 * @brief Structure to hold both residual block ID and cost function
 */
struct ObservationInfo {
    ceres::ResidualBlockId residual_id;
    factor::MonoPnPFactor* cost_function;
    
    ObservationInfo(ceres::ResidualBlockId id, factor::MonoPnPFactor* func)
        : residual_id(id), cost_function(func) {}
};

/**
 * @brief Pose optimizer for VIO system using Ceres
 */
class PoseOptimizer {
public:
    /**
     * @brief Constructor
     */
    PoseOptimizer();
    
    /**
     * @brief Destructor
     */
    ~PoseOptimizer() = default;
    
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
};

} // namespace lightweight_vio
