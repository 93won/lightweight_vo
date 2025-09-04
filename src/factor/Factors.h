#pragma once

#include <ceres/cost_function.h>
#include <ceres/sized_cost_function.h>
#include <Eigen/Dense>
#include <sophus/se3.hpp>

// Define Vector6d as it's not available in standard Eigen
namespace Eigen {
    typedef Matrix<double, 6, 1> Vector6d;
}

namespace lightweight_vio {
namespace factor {

/**
 * @brief Camera parameters structure
 */
struct CameraParameters {
    double fx, fy, cx, cy;
    
    CameraParameters(double fx_, double fy_, double cx_, double cy_)
        : fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}
};

/**
 * @brief Monocular PnP cost function with analytical Jacobian
 * Uses Twb (body-to-world) pose with T_CB (body-to-camera) extrinsics
 * Residual: observed_pixel - projected_pixel
 * Parameters: SE3 pose in tangent space [6] (Twb)
 * Residual dimension: [2]
 */
class PnPFactor : public ceres::SizedCostFunction<2, 6> {
public:
    /**
     * @brief Constructor
     * @param observation Observed 2D pixel coordinates [u, v]
     * @param world_point 3D point in world coordinates
     * @param camera_params Camera intrinsic parameters
     * @param Tcb Body-to-camera transformation matrix [4x4] (T_CB)
     * @param information Information matrix (precision matrix) [2x2]
     */
    PnPFactor(const Eigen::Vector2d& observation,
                  const Eigen::Vector3d& world_point,
                  const CameraParameters& camera_params,
                  const Eigen::Matrix4d& Tcb,
                  const Eigen::Matrix2d& information = Eigen::Matrix2d::Identity());

    /**
     * @brief Set outlier flag to disable optimization for this factor
     * @param is_outlier If true, this factor will not contribute to optimization
     */
    void set_outlier(bool is_outlier) { m_is_outlier = is_outlier; }
    
    /**
     * @brief Get outlier flag
     * @return true if this factor is marked as outlier
     */
    bool is_outlier() const { return m_is_outlier; }

    /**
     * @brief Evaluate residual and Jacobian
     * @param parameters SE3 pose parameters in tangent space [6]
     * @param residuals Output residual [2]
     * @param jacobians Output Jacobian matrices [2x6] if not nullptr
     * @return true if evaluation successful
     */
    virtual bool Evaluate(double const* const* parameters,
                         double* residuals,
                         double** jacobians) const override;

    /**
     * @brief Compute Chi-square error for outlier detection
     * @param parameters SE3 pose parameters in tangent space [6]
     * @return Chi-square error value
     */
    double compute_chi_square(double const* const* parameters) const;

private:
    Eigen::Vector2d m_observation;    // Observed pixel coordinates
    Eigen::Vector3d m_world_point;    // 3D world coordinates
    CameraParameters m_camera_params; // Camera intrinsics
    Eigen::Matrix4d m_Tcb;            // Body-to-camera transformation (T_CB)
    Eigen::Matrix2d m_information;    // Information matrix (precision matrix)
    bool m_is_outlier;                // Outlier flag to disable optimization
};

/**
 * @brief Bundle Adjustment monocular cost function with analytical Jacobian
 * Uses Twb (body-to-world) pose with T_CB (body-to-camera) extrinsics
 * Residual: observed_pixel - projected_pixel
 * Parameters: SE3 pose in tangent space [6] (Twb), 3D point position [3]
 * Residual dimension: [2]
 */
class BAFactor : public ceres::SizedCostFunction<2, 6, 3> {
public:
    /**
     * @brief Constructor
     * @param observation Observed 2D pixel coordinates [u, v]
     * @param camera_params Camera intrinsic parameters
     * @param Tcb Body-to-camera transformation matrix [4x4] (T_CB)
     * @param information Information matrix (precision matrix) [2x2]
     */
    BAFactor(const Eigen::Vector2d& observation,
             const CameraParameters& camera_params,
             const Eigen::Matrix4d& Tcb,
             const Eigen::Matrix2d& information = Eigen::Matrix2d::Identity());

    /**
     * @brief Set outlier flag to disable optimization for this factor
     * @param is_outlier If true, this factor will not contribute to optimization
     */
    void set_outlier(bool is_outlier) { m_is_outlier = is_outlier; }
    
    /**
     * @brief Get outlier flag
     * @return true if this factor is marked as outlier
     */
    bool is_outlier() const { return m_is_outlier; }

    /**
     * @brief Evaluate residual and Jacobian
     * @param parameters[0] SE3 pose parameters in tangent space [6] (Twb)
     * @param parameters[1] 3D point position in world coordinates [3]
     * @param residuals Output residual [2]
     * @param jacobians[0] Output Jacobian w.r.t pose [2x6] if not nullptr
     * @param jacobians[1] Output Jacobian w.r.t point [2x3] if not nullptr
     * @return true if evaluation successful
     */
    virtual bool Evaluate(double const* const* parameters,
                         double* residuals,
                         double** jacobians) const override;

    /**
     * @brief Compute Chi-square error for outlier detection
     * @param parameters[0] SE3 pose parameters in tangent space [6]
     * @param parameters[1] 3D point position in world coordinates [3]
     * @return Chi-square error value
     */
    double compute_chi_square(double const* const* parameters) const;

private:
    Eigen::Vector2d m_observation;    // Observed pixel coordinates
    CameraParameters m_camera_params; // Camera intrinsics
    Eigen::Matrix4d m_Tcb;            // Body-to-camera transformation (T_CB)
    Eigen::Matrix2d m_information;    // Information matrix (precision matrix)
    bool m_is_outlier;                // Outlier flag to disable optimization
};

} // namespace factor
} // namespace lightweight_vio
