#pragma once

#include <ceres/cost_function.h>
#include <ceres/sized_cost_function.h>
#include <Eigen/Dense>
#include <sophus/se3.hpp>

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
 * Residual: observed_pixel - projected_pixel
 * Parameters: SE3 pose in tangent space [6]
 * Residual dimension: [2]
 */
class MonoPnPFactor : public ceres::SizedCostFunction<2, 6> {
public:
    /**
     * @brief Constructor
     * @param observation Observed 2D pixel coordinates [u, v]
     * @param world_point 3D point in world coordinates
     * @param camera_params Camera intrinsic parameters
     * @param information Information matrix (precision matrix) [2x2]
     */
    MonoPnPFactor(const Eigen::Vector2d& observation,
                  const Eigen::Vector3d& world_point,
                  const CameraParameters& camera_params,
                  const Eigen::Matrix2d& information = Eigen::Matrix2d::Identity());

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
    Eigen::Matrix2d m_information;    // Information matrix (precision matrix)
};

} // namespace factor
} // namespace lightweight_vio
