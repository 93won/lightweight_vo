#pragma once

#include <ceres/local_parameterization.h>
#include <Eigen/Dense>
#include <sophus/se3.hpp>

// Define Vector6d as it's not available in standard Eigen
namespace Eigen {
    typedef Matrix<double, 6, 1> Vector6d;
}

namespace lightweight_vio {
namespace factor {

/**
 * @brief SE3 Local Parameterization for Ceres optimization
 * Parameterizes SE3 group using 6DoF tangent space representation
 * Parameters: [so3_x, so3_y, so3_z, t_x, t_y, t_z]
 * 
 * For Twb (body to world transform), we use right multiplication:
 * Twb_new = Twb * exp(delta)
 * 
 * This means the perturbation is applied in the body frame.
 */
class SE3LocalParameterization : public ceres::LocalParameterization {
public:
    SE3LocalParameterization() = default;
    virtual ~SE3LocalParameterization() = default;

    /**
     * @brief Plus operation: x_plus_delta = SE3(x) * exp(delta)
     * @param x Current SE3 parameters in tangent space [6]
     * @param delta Update vector in tangent space [6] 
     * @param x_plus_delta Updated SE3 parameters [6]
     * @return true if successful
     */
    virtual bool Plus(const double* x,
                     const double* delta,
                     double* x_plus_delta) const override;
    
    /**
     * @brief Compute Jacobian of Plus operation w.r.t delta
     * @param x Current parameters [6]
     * @param jacobian Output jacobian matrix [6x6] in row-major order
     * @return true if successful
     */
    virtual bool ComputeJacobian(const double* x,
                                double* jacobian) const override;
    
    /**
     * @brief Global size of the parameter (tangent space dimension)
     */
    virtual int GlobalSize() const override { return 6; }
    
    /**
     * @brief Local size of the perturbation (tangent space dimension)
     */
    virtual int LocalSize() const override { return 6; }

private:
    /**
     * @brief Convert SE3 tangent space vector to SE3 group element
     * @param tangent SE3 tangent space vector [so3, translation]
     * @return SE3 group element
     */
    static Sophus::SE3d TangentToSE3(const Eigen::Vector6d& tangent);

    /**
     * @brief Convert SE3 group element to tangent space vector
     * @param se3 SE3 group element
     * @return SE3 tangent space vector [so3, translation]
     */
    static Eigen::Vector6d SE3ToTangent(const Sophus::SE3d& se3);
};

} // namespace factor
} // namespace lightweight_vio
