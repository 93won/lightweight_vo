#include "Parameters.h"

namespace lightweight_vio {
namespace factor {

bool SE3LocalParameterization::Plus(const double* x,
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

bool SE3LocalParameterization::ComputeJacobian(const double* x,
                                              double* jacobian) const {
    try {
        Eigen::Map<const Eigen::Vector6d> tangent(x);
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> jac(jacobian);
        
        // For right perturbation: Twb_new = Twb * exp(delta)
        // The Jacobian is the right Jacobian of SE3
        Sophus::SE3d se3 = TangentToSE3(tangent);
        
        // Right Jacobian for SE3
        // For small perturbations, we can use the adjoint representation
        jac = se3.Adj();
        
        return true;
    } catch (const std::exception& e) {
        // Fallback to identity for numerical stability
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> jac(jacobian);
        jac.setIdentity();
        return true;
    }
}

Sophus::SE3d SE3LocalParameterization::TangentToSE3(const Eigen::Vector6d& tangent) {
    // Tangent vector format: [so3_x, so3_y, so3_z, t_x, t_y, t_z]
    Eigen::Vector3d so3_part = tangent.head<3>();
    Eigen::Vector3d translation_part = tangent.tail<3>();
    
    // Create SE3 from rotation and translation
    Sophus::SO3d rotation = Sophus::SO3d::exp(so3_part);
    return Sophus::SE3d(rotation, translation_part);
}

Eigen::Vector6d SE3LocalParameterization::SE3ToTangent(const Sophus::SE3d& se3) {
    Eigen::Vector6d tangent;
    
    // Extract rotation (as so3 tangent) and translation
    tangent.head<3>() = se3.so3().log();
    tangent.tail<3>() = se3.translation();
    
    return tangent;
}

} // namespace factor
} // namespace lightweight_vio
