#include "SE3Manifolds.h"

namespace lightweight_vio {
namespace factor {

// SE3Manifold implementation
bool SE3Manifold::Plus(const double* x,
                       const double* delta,
                       double* x_plus_delta) const {
    try {
        // Convert arrays to Eigen vectors
        Eigen::Map<const Eigen::Vector6d> current_tangent(x);
        Eigen::Map<const Eigen::Vector6d> delta_tangent(delta);
        Eigen::Map<Eigen::Vector6d> result_tangent(x_plus_delta);
        
        // Convert current tangent to SE3
        Sophus::SE3d current_se3 = TangentToSE3(current_tangent);
        
        // Apply delta as left multiplication: exp(delta) * current
        Sophus::SE3d delta_se3 = Sophus::SE3d::exp(delta_tangent);
        Sophus::SE3d result_se3 = delta_se3 * current_se3;
        
        // Convert back to tangent space
        result_tangent = SE3ToTangent(result_se3);
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool SE3Manifold::PlusJacobian(const double* x, double* jacobian) const {
    try {
        Eigen::Map<const Eigen::Vector6d> tangent(x);
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> jac(jacobian);
        
        // For SE3, the left Jacobian is approximately identity for small perturbations
        // More precisely, it should be the left Jacobian of SE3
        Sophus::SE3d se3 = TangentToSE3(tangent);
        jac = Sophus::SE3d::leftJacobian(tangent);
        
        return true;
    } catch (const std::exception& e) {
        // Fallback to identity
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> jac(jacobian);
        jac.setIdentity();
        return true;
    }
}

bool SE3Manifold::Minus(const double* y,
                        const double* x,
                        double* y_minus_x) const {
    try {
        Eigen::Map<const Eigen::Vector6d> y_tangent(y);
        Eigen::Map<const Eigen::Vector6d> x_tangent(x);
        Eigen::Map<Eigen::Vector6d> diff_tangent(y_minus_x);
        
        // Convert to SE3
        Sophus::SE3d y_se3 = TangentToSE3(y_tangent);
        Sophus::SE3d x_se3 = TangentToSE3(x_tangent);
        
        // Compute difference: x^{-1} * y
        Sophus::SE3d diff_se3 = x_se3.inverse() * y_se3;
        diff_tangent = diff_se3.log();
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool SE3Manifold::MinusJacobian(const double* x, double* jacobian) const {
    try {
        Eigen::Map<const Eigen::Vector6d> tangent(x);
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> jac(jacobian);
        
        // Right Jacobian inverse for SE3
        jac = Sophus::SE3d::leftJacobianInverse(tangent);
        
        return true;
    } catch (const std::exception& e) {
        // Fallback to identity
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> jac(jacobian);
        jac.setIdentity();
        return true;
    }
}

Sophus::SE3d SE3Manifold::TangentToSE3(const Eigen::Vector6d& tangent) {
    return Sophus::SE3d::exp(tangent);
}

Eigen::Vector6d SE3Manifold::SE3ToTangent(const Sophus::SE3d& se3) {
    return se3.log();
}

// SO3Manifold implementation
bool SO3Manifold::Plus(const double* x,
                       const double* delta,
                       double* x_plus_delta) const {
    try {
        Eigen::Map<const Eigen::Vector3d> current_tangent(x);
        Eigen::Map<const Eigen::Vector3d> delta_tangent(delta);
        Eigen::Map<Eigen::Vector3d> result_tangent(x_plus_delta);
        
        Sophus::SO3d current_so3 = TangentToSO3(current_tangent);
        Sophus::SO3d delta_so3 = Sophus::SO3d::exp(delta_tangent);
        Sophus::SO3d result_so3 = delta_so3 * current_so3;
        
        result_tangent = SO3ToTangent(result_so3);
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool SO3Manifold::PlusJacobian(const double* x, double* jacobian) const {
    try {
        Eigen::Map<const Eigen::Vector3d> tangent(x);
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jac(jacobian);
        
        jac = Sophus::SO3d::leftJacobian(tangent);
        
        return true;
    } catch (const std::exception& e) {
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jac(jacobian);
        jac.setIdentity();
        return true;
    }
}

bool SO3Manifold::Minus(const double* y,
                        const double* x,
                        double* y_minus_x) const {
    try {
        Eigen::Map<const Eigen::Vector3d> y_tangent(y);
        Eigen::Map<const Eigen::Vector3d> x_tangent(x);
        Eigen::Map<Eigen::Vector3d> diff_tangent(y_minus_x);
        
        Sophus::SO3d y_so3 = TangentToSO3(y_tangent);
        Sophus::SO3d x_so3 = TangentToSO3(x_tangent);
        
        Sophus::SO3d diff_so3 = x_so3.inverse() * y_so3;
        diff_tangent = diff_so3.log();
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool SO3Manifold::MinusJacobian(const double* x, double* jacobian) const {
    try {
        Eigen::Map<const Eigen::Vector3d> tangent(x);
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jac(jacobian);
        
        jac = Sophus::SO3d::leftJacobianInverse(tangent);
        
        return true;
    } catch (const std::exception& e) {
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jac(jacobian);
        jac.setIdentity();
        return true;
    }
}

Sophus::SO3d SO3Manifold::TangentToSO3(const Eigen::Vector3d& tangent) {
    return Sophus::SO3d::exp(tangent);
}

Eigen::Vector3d SO3Manifold::SO3ToTangent(const Sophus::SO3d& so3) {
    return so3.log();
}

} // namespace factor
} // namespace lightweight_vio
