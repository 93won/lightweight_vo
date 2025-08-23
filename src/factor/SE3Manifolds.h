#pragma once

#include <ceres/manifold.h>
#include <Eigen/Dense>
#include <sophus/se3.hpp>

// Define Vector6d as it's not available in standard Eigen
namespace Eigen {
    typedef Matrix<double, 6, 1> Vector6d;
}

namespace lightweight_vio {
namespace factor {

/**
 * @brief SE3 Manifold for Ceres optimization
 * Parameterizes SE3 group using 6DoF tangent space representation
 * Parameters: [so3_x, so3_y, so3_z, t_x, t_y, t_z]
 */
class SE3Manifold : public ceres::Manifold {
public:
    SE3Manifold() = default;
    virtual ~SE3Manifold() = default;

    /**
     * @brief Plus operation: x_plus_delta = SE3::exp(delta) * SE3::exp(x)
     * @param x Current SE3 parameters in tangent space [6]
     * @param delta Update vector in tangent space [6] 
     * @param x_plus_delta Updated SE3 parameters [6]
     * @return true if successful
     */
    bool Plus(const double* x,
              const double* delta,
              double* x_plus_delta) const override;

    /**
     * @brief Jacobian of Plus operation w.r.t delta
     * @param x Current parameters [6]
     * @param jacobian Output jacobian matrix [6x6] in row-major order
     * @return true if successful
     */
    bool PlusJacobian(const double* x, double* jacobian) const override;

    /**
     * @brief Compute the difference between two SE3 parameters
     * @param y Target SE3 parameters [6]
     * @param x Current SE3 parameters [6] 
     * @param y_minus_x Output difference in tangent space [6]
     * @return true if successful
     */
    bool Minus(const double* y,
               const double* x,
               double* y_minus_x) const override;

    /**
     * @brief Jacobian of Minus operation w.r.t first argument
     * @param y Target parameters [6]
     * @param x Current parameters [6]
     * @param jacobian Output jacobian matrix [6x6] in row-major order
     * @return true if successful
     */
    bool MinusJacobian(const double* x, double* jacobian) const override;

    int AmbientSize() const override { return 6; }
    int TangentSize() const override { return 6; }

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

/**
 * @brief SO3 Manifold for rotation-only optimization
 * Parameterizes SO3 group using 3DoF axis-angle representation
 */
class SO3Manifold : public ceres::Manifold {
public:
    SO3Manifold() = default;
    virtual ~SO3Manifold() = default;

    bool Plus(const double* x,
              const double* delta,
              double* x_plus_delta) const override;

    bool PlusJacobian(const double* x, double* jacobian) const override;

    bool Minus(const double* y,
               const double* x,
               double* y_minus_x) const override;

    bool MinusJacobian(const double* x, double* jacobian) const override;

    int AmbientSize() const override { return 3; }
    int TangentSize() const override { return 3; }

private:
    static Sophus::SO3d TangentToSO3(const Eigen::Vector3d& tangent);
    static Eigen::Vector3d SO3ToTangent(const Sophus::SO3d& so3);
};

} // namespace factor
} // namespace lightweight_vio
