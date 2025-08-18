#include "PnPFactors.h"
#include <limits>

namespace lightweight_vio {
namespace factor {

// MonoPnPFactor implementation
MonoPnPFactor::MonoPnPFactor(const Eigen::Vector2d& observation,
                            const Eigen::Vector3d& world_point,
                            const CameraParameters& camera_params,
                            const Eigen::Matrix2d& information)
    : m_observation(observation), m_world_point(world_point), 
      m_camera_params(camera_params), m_information(information) {}

bool MonoPnPFactor::Evaluate(double const* const* parameters,
                             double* residuals,
                             double** jacobians) const {
    // Extract SE3 parameters from tangent space
    Eigen::Map<const Eigen::Vector6d> se3_tangent(parameters[0]);
    
    // Convert to SE3 and transform point to camera coordinates
    Sophus::SE3d pose = Sophus::SE3d::exp(se3_tangent);
    Eigen::Vector3d point_camera = pose * m_world_point;
    
    double x = point_camera.x();
    double y = point_camera.y();
    double z = point_camera.z();
    
    // Check for valid depth
    if (z <= 1e-6) {
        return false;
    }
    
    // Project to image plane
    double u = m_camera_params.fx * x / z + m_camera_params.cx;
    double v = m_camera_params.fy * y / z + m_camera_params.cy;
    
    // Compute residuals: observation - projection
    Eigen::Vector2d residual_vec;
    residual_vec << m_observation.x() - u, m_observation.y() - v;
    
    // Apply information matrix weighting: r_weighted = sqrt(Information) * r
    Eigen::LLT<Eigen::Matrix2d> llt(m_information);
    if (llt.info() == Eigen::Success) {
        // Use Cholesky decomposition: Information = L * L^T
        // Weighted residual = L * residual
        Eigen::Vector2d weighted_residual = llt.matrixL() * residual_vec;
        residuals[0] = weighted_residual[0];
        residuals[1] = weighted_residual[1];
    } else {
        // Fallback to unweighted if Cholesky fails
        residuals[0] = residual_vec[0];
        residuals[1] = residual_vec[1];
    }
    
    // Compute analytical jacobians if requested
    if (jacobians && jacobians[0]) {
        Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jac(jacobians[0]);
        
        double z_inv = 1.0 / z;
        double z_inv_sq = z_inv * z_inv;
        
        // Jacobian of projection w.r.t camera coordinates [2x3]
        Eigen::Matrix<double, 2, 3> J_proj_camera;
        J_proj_camera << 
            m_camera_params.fx * z_inv,  0.0,                          -m_camera_params.fx * x * z_inv_sq,
            0.0,                         m_camera_params.fy * z_inv,   -m_camera_params.fy * y * z_inv_sq;
        
        // Jacobian of camera coordinates w.r.t SE3 tangent space [3x6]
        Eigen::Matrix<double, 3, 6> J_camera_se3;
        
        // Rotation part: d(R*p)/d(so3) = -R * hat(p)
        Eigen::Matrix3d R = pose.rotationMatrix();
        J_camera_se3.block<3, 3>(0, 0) = -R * Sophus::SO3d::hat(m_world_point);
        
        // Translation part: d(R*p + t)/d(t) = R
        J_camera_se3.block<3, 3>(0, 3) = R;
        
        // Chain rule: J = -J_proj_camera * J_camera_se3
        Eigen::Matrix<double, 2, 6> unweighted_jac = -J_proj_camera * J_camera_se3;
        
        // Apply information matrix weighting to Jacobian
        Eigen::LLT<Eigen::Matrix2d> llt(m_information);
        if (llt.info() == Eigen::Success) {
            jac = llt.matrixL() * unweighted_jac;
        } else {
            jac = unweighted_jac;
        }
    }
    
    return true;
}

double MonoPnPFactor::compute_chi_square(double const* const* parameters) const {
    // Extract SE3 parameters from tangent space
    Eigen::Map<const Eigen::Vector6d> se3_tangent(parameters[0]);
    
    // Convert to SE3 and transform point to camera coordinates
    Sophus::SE3d pose = Sophus::SE3d::exp(se3_tangent);
    Eigen::Vector3d point_camera = pose * m_world_point;
    
    double x = point_camera.x();
    double y = point_camera.y();
    double z = point_camera.z();
    
    // Check for valid depth
    if (z <= 1e-6) {
        return std::numeric_limits<double>::max(); // Invalid, return large chi-square
    }
    
    // Project to image plane
    double u = m_camera_params.fx * x / z + m_camera_params.cx;
    double v = m_camera_params.fy * y / z + m_camera_params.cy;
    
    // Compute residuals: observation - projection
    Eigen::Vector2d residual_vec;
    residual_vec << m_observation.x() - u, m_observation.y() - v;
    
    // Chi-square error with information matrix: r^T * Information * r
    double chi2_error = residual_vec.transpose() * m_information * residual_vec;
    
    return chi2_error;
}

} // namespace factor
} // namespace lightweight_vio
