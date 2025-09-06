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
 * @brief SE3 Global Parameterization for Ceres optimization
 * Parameterizes SE3 group using 6DoF tangent space representation
 * Parameters: [t_x, t_y, t_z, so3_x, so3_y, so3_z] (Ceres order)
 * 
 * For Twb (body to world transform), we use right multiplication:
 * Twb_new = Twb * exp(delta)
 * 
 * This means the perturbation is applied in the body frame.
 */
class SE3GlobalParameterization : public ceres::LocalParameterization {
public:
    SE3GlobalParameterization() : m_is_fixed(false) {}
    virtual ~SE3GlobalParameterization() = default;

    /**
     * @brief Set parameter as fixed (prevents updates during optimization)
     * @param is_fixed If true, parameters will not be updated
     */
    void set_fixed(bool is_fixed) { m_is_fixed = is_fixed; }
    
    /**
     * @brief Check if parameter is fixed
     * @return true if parameter is fixed
     */
    bool is_fixed() const { return m_is_fixed; }

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
     * @param tangent SE3 tangent space vector [translation, so3] (Ceres order)
     * @return SE3 group element
     */
    static Sophus::SE3d TangentToSE3(const Eigen::Vector6d& tangent);

    /**
     * @brief Convert SE3 group element to tangent space vector
     * @param se3 SE3 group element
     * @return SE3 tangent space vector [translation, so3] (Ceres order)
     */
    static Eigen::Vector6d SE3ToTangent(const Sophus::SE3d& se3);

    bool m_is_fixed;  // Flag to prevent parameter updates when true
};

/**
 * @brief MapPoint (3D Point) Parameterization for Ceres optimization
 * Parameterizes 3D points in world coordinates using standard Euclidean parameterization
 * Parameters: [x, y, z] (world coordinates)
 * 
 * This is a simple identity parameterization for 3D points, but provides
 * a consistent interface for future extensions (e.g., inverse depth parameterization)
 */
class MapPointParameterization : public ceres::LocalParameterization {
public:
    MapPointParameterization() : m_is_fixed(false) {}
    virtual ~MapPointParameterization() = default;

    /**
     * @brief Set parameter as fixed (prevents updates during optimization)
     * @param is_fixed If true, parameters will not be updated
     */
    void set_fixed(bool is_fixed) { m_is_fixed = is_fixed; }
    
    /**
     * @brief Check if parameter is fixed
     * @return true if parameter is fixed
     */
    bool is_fixed() const { return m_is_fixed; }

    /**
     * @brief Plus operation: x_plus_delta = x + delta
     * @param x Current 3D point coordinates [3]
     * @param delta Update vector [3] 
     * @param x_plus_delta Updated 3D point coordinates [3]
     * @return true if successful
     */
    virtual bool Plus(const double* x,
                     const double* delta,
                     double* x_plus_delta) const override;
    
    /**
     * @brief Compute Jacobian of Plus operation w.r.t delta
     * @param x Current parameters [3]
     * @param jacobian Output jacobian matrix [3x3] in row-major order
     * @return true if successful
     */
    virtual bool ComputeJacobian(const double* x,
                                double* jacobian) const override;
    
    /**
     * @brief Global size of the parameter (3D point dimension)
     */
    virtual int GlobalSize() const override { return 3; }
    
    /**
     * @brief Local size of the perturbation (3D point dimension)
     */
    virtual int LocalSize() const override { return 3; }

private:
    bool m_is_fixed;  // Flag to prevent parameter updates when true
};

} // namespace factor
} // namespace lightweight_vio
