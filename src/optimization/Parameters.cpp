/**
 * @file      Parameters.cpp
 * @brief     Implements parameter block management for Ceres optimization.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-08-18
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "optimization/Parameters.h"

namespace lightweight_vio {
namespace factor {

bool SE3GlobalParameterization::Plus(const double* x,
                                   const double* delta,
                                   double* x_plus_delta) const {
    try {
        // If parameter is fixed, don't apply any updates
        if (m_is_fixed) {
            // Copy original parameters without applying delta
            for (int i = 0; i < 6; ++i) {
                x_plus_delta[i] = x[i];
            }
            return true;
        }
        
        // Convert arrays to Eigen vectors
        Eigen::Map<const Eigen::Vector6d> current_tangent(x);
        Eigen::Map<const Eigen::Vector6d> delta_tangent(delta);
        Eigen::Map<Eigen::Vector6d> result_tangent(x_plus_delta);
        
        // Convert current tangent to SE3
        Sophus::SE3d current_se3 = TangentToSE3(current_tangent);
        
        // Apply delta as right multiplication: current * exp(delta)
        // This is appropriate for Twb (body to world) where perturbation is in body frame
        Sophus::SE3d delta_se3 = Sophus::SE3d::exp(delta_tangent);
        Sophus::SE3d result_se3 = current_se3 * delta_se3;
        
        // Convert back to tangent space
        result_tangent = SE3ToTangent(result_se3);
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool SE3GlobalParameterization::ComputeJacobian(const double* x,
                                              double* jacobian) const {
    // For small perturbations in SE3, the Jacobian can be approximated as Identity
    // This is much faster than computing the exact right Jacobian
    Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> jac(jacobian);
    jac.setIdentity();
    return true;
}

Sophus::SE3d SE3GlobalParameterization::TangentToSE3(const Eigen::Vector6d& tangent) {
    // Ceres order: [t_x, t_y, t_z, so3_x, so3_y, so3_z]
    // Use Sophus SE3::exp for consistent parameterization with V matrix
    return Sophus::SE3d::exp(tangent);
}

Eigen::Vector6d SE3GlobalParameterization::SE3ToTangent(const Sophus::SE3d& se3) {
    // Use SE3::log() for consistency with SE3::exp() in TangentToSE3()
    // This ensures proper V matrix handling
    return se3.log();
}

bool MapPointParameterization::Plus(const double* x,
                                  const double* delta,
                                  double* x_plus_delta) const {
    try {
        // If parameter is fixed, don't apply any updates
        if (m_is_fixed) {
            // Copy original parameters without applying delta
            x_plus_delta[0] = x[0];  // x coordinate
            x_plus_delta[1] = x[1];  // y coordinate
            x_plus_delta[2] = x[2];  // z coordinate
            return true;
        }
        
        // Simple Euclidean addition for 3D points
        // x_plus_delta = x + delta
        x_plus_delta[0] = x[0] + delta[0];  // x coordinate
        x_plus_delta[1] = x[1] + delta[1];  // y coordinate
        x_plus_delta[2] = x[2] + delta[2];  // z coordinate
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool MapPointParameterization::ComputeJacobian(const double* x,
                                             double* jacobian) const {
    // For Euclidean 3D points, the Jacobian of Plus operation w.r.t delta is Identity
    // d(x + delta)/d(delta) = I (3x3 identity matrix)
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jac(jacobian);
    jac.setIdentity();
    return true;
}

// ===============================================================================
// VELOCITY PARAMETERIZATION IMPLEMENTATION
// ===============================================================================

bool VelocityParameterization::Plus(const double* x,
                                   const double* delta,
                                   double* x_plus_delta) const {
    try {
        // If parameter is fixed, don't apply any updates
        if (m_is_fixed) {
            // Copy original parameters without applying delta
            for (int i = 0; i < 3; ++i) {
                x_plus_delta[i] = x[i];
            }
            return true;
        }
        
        // Simple addition for Euclidean velocity parameters
        for (int i = 0; i < 3; ++i) {
            x_plus_delta[i] = x[i] + delta[i];
        }
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool VelocityParameterization::ComputeJacobian(const double* x,
                                              double* jacobian) const {
    // For Euclidean 3D velocity, the Jacobian of Plus operation w.r.t delta is Identity
    // d(x + delta)/d(delta) = I (3x3 identity matrix)
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jac(jacobian);
    jac.setIdentity();
    return true;
}

// ===============================================================================
// BIAS PARAMETERIZATION IMPLEMENTATION
// ===============================================================================

bool BiasParameterization::Plus(const double* x,
                               const double* delta,
                               double* x_plus_delta) const {
    try {
        // If parameter is fixed, don't apply any updates
        if (m_is_fixed) {
            // Copy original parameters without applying delta
            for (int i = 0; i < 3; ++i) {
                x_plus_delta[i] = x[i];
            }
            return true;
        }
        
        // Simple addition for Euclidean bias parameters
        for (int i = 0; i < 3; ++i) {
            x_plus_delta[i] = x[i] + delta[i];
        }
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool BiasParameterization::ComputeJacobian(const double* x,
                                          double* jacobian) const {
    // For Euclidean 3D bias, the Jacobian of Plus operation w.r.t delta is Identity
    // d(x + delta)/d(delta) = I (3x3 identity matrix)
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jac(jacobian);
    jac.setIdentity();
    return true;
}

} // namespace factor
} // namespace lightweight_vio
