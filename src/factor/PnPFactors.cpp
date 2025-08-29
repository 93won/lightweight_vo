#include "PnPFactors.h"
#include <limits>

namespace lightweight_vio {
namespace factor {

// MonoPnPFactor implementation
MonoPnPFactor::MonoPnPFactor(const Eigen::Vector2d& observation,
                            const Eigen::Vector3d& world_point,
                            const CameraParameters& camera_params,
                            const Eigen::Matrix4d& Tcb,
                            const Eigen::Matrix2d& information)
    : m_observation(observation), m_world_point(world_point), 
      m_camera_params(camera_params), m_Tcb(Tcb), m_information(information), m_is_outlier(false) {}

bool MonoPnPFactor::Evaluate(double const* const* parameters,
                             double* residuals,
                             double** jacobians) const {
    // If marked as outlier, set residuals to zero and jacobians to zero
    if (m_is_outlier) {
        residuals[0] = 0.0;
        residuals[1] = 0.0;
        
        if (jacobians && jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jac(jacobians[0]);
            jac.setZero();
        }
        return true;
    }

    // Extract SE3 parameters from tangent space (Twb)
    Eigen::Map<const Eigen::Vector6d> se3_tangent(parameters[0]);
    
    // Convert to SE3 pose (Twb)
    Sophus::SE3d Twb = Sophus::SE3d::exp(se3_tangent);
    
    // g2o 코드 참고: Twb를 사용하여 Tcw 계산
    Eigen::Matrix3d Rwb = Twb.rotationMatrix();
    Eigen::Vector3d twb = Twb.translation();
    Eigen::Matrix3d Rbw = Rwb.transpose();
    Eigen::Vector3d tbw = -Rbw * twb;
    
    // Tcw = Tcb * Tbw
    Eigen::Matrix3d Rcw = m_Tcb.block<3, 3>(0, 0) * Rbw;
    Eigen::Vector3d tcw = m_Tcb.block<3, 3>(0, 0) * tbw + m_Tcb.block<3, 1>(0, 3);
    
    // Transform world point to camera coordinates: Pc = Rcw * Pw + tcw
    Eigen::Vector3d point_camera = Rcw * m_world_point + tcw;
    
    double x = point_camera.x();
    double y = point_camera.y();
    double z = point_camera.z();
    
    // Check for valid depth
    if (z <= 1e-6) {
        return false;
    }
    
    double z_inv = 1.0 / z;
    
    // Project to image plane
    double u = m_camera_params.fx * x * z_inv + m_camera_params.cx;
    double v = m_camera_params.fy * y * z_inv + m_camera_params.cy;
    
    // Compute residuals: observation - projection
    Eigen::Vector2d residual_vec;
    residual_vec << m_observation.x() - u, m_observation.y() - v;
    
    // Apply information matrix weighting to residuals: r_weighted = sqrt(Info) * r
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
        
        double z_inv_sq = z_inv * z_inv;
        
        // g2o 코드 참고: Jacobian 계산
        // Tcb의 회전 부분 캐싱
        Eigen::Matrix3d Rcb = m_Tcb.block<3, 3>(0, 0);
        
        // Tbw에서의 body 좌표: Pb = Rbw * Pw + tbw
        Eigen::Vector3d Pb = Rbw * m_world_point + tbw;
        
        // 카메라 투영 Jacobian 계산
        Eigen::Matrix<double, 2, 3> J_proj_camera;
        J_proj_camera(0, 0) = -m_camera_params.fx * z_inv;
        J_proj_camera(0, 1) = 0.0;
        J_proj_camera(0, 2) = x * m_camera_params.fx * z_inv_sq;
        J_proj_camera(1, 0) = 0.0;
        J_proj_camera(1, 1) = -m_camera_params.fy * z_inv;
        J_proj_camera(1, 2) = y * m_camera_params.fy * z_inv_sq;
        
        // g2o 코드 참고: Twb에 대한 Jacobian
        Eigen::Matrix<double, 2, 3> JdPwb = J_proj_camera * (-Rcb);
        Eigen::Matrix3d hatPb = Sophus::SO3d::hat(Pb);
        Eigen::Matrix<double, 2, 3> JdRwb = J_proj_camera * Rcb * hatPb;
        
        // 전체 Jacobian 조합 [rotation | translation]
        Eigen::Matrix<double, 2, 6> unweighted_jac;
        unweighted_jac << JdRwb, JdPwb;
        
        // Apply information matrix weighting to Jacobian: J_weighted = sqrt(Info) * J
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
    // Extract SE3 parameters from tangent space (Twb)
    Eigen::Map<const Eigen::Vector6d> se3_tangent(parameters[0]);
    
    // Convert to SE3 pose (Twb)
    Sophus::SE3d Twb = Sophus::SE3d::exp(se3_tangent);
    
    // g2o 코드 참고: Twb를 사용하여 Tcw 계산
    Eigen::Matrix3d Rwb = Twb.rotationMatrix();
    Eigen::Vector3d twb = Twb.translation();
    Eigen::Matrix3d Rbw = Rwb.transpose();
    Eigen::Vector3d tbw = -Rbw * twb;
    
    // Tcw = Tcb * Tbw
    Eigen::Matrix3d Rcw = m_Tcb.block<3, 3>(0, 0) * Rbw;
    Eigen::Vector3d tcw = m_Tcb.block<3, 3>(0, 0) * tbw + m_Tcb.block<3, 1>(0, 3);
    
    // Transform world point to camera coordinates: Pc = Rcw * Pw + tcw
    Eigen::Vector3d point_camera = Rcw * m_world_point + tcw;
    
    double x = point_camera.x();
    double y = point_camera.y();
    double z = point_camera.z();
    
    // Check for valid depth
    if (z <= 1e-6) {
        return std::numeric_limits<double>::max(); // Invalid, return large chi-square
    }
    
    double z_inv = 1.0 / z;
    
    // Project to image plane
    double u = m_camera_params.fx * x * z_inv + m_camera_params.cx;
    double v = m_camera_params.fy * y * z_inv + m_camera_params.cy;
    
    // Compute residuals: observation - projection
    Eigen::Vector2d residual_vec;
    residual_vec << m_observation.x() - u, m_observation.y() - v;
    
    // Chi-square error with information matrix: r^T * Information * r
    double chi2_error = residual_vec.transpose() * m_information * residual_vec;
    
    return chi2_error;
}

} // namespace factor
} // namespace lightweight_vio
