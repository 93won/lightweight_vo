#include "Parameters.h"

namespace lightweight_vio {
namespace factor {

bool SE3GlobalParameterization::Plus(const double* x,
                                   const double* delta,
                                   double* x_plus_delta) const {
    try {
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

} // namespace factor
} // namespace lightweight_vio
