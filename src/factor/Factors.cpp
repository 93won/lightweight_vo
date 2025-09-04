#include "Factors.h"
#include <limits>

namespace lightweight_vio {
namespace factor {

// PnPFactor implementation
PnPFactor::PnPFactor(const Eigen::Vector2d& observation,
                            const Eigen::Vector3d& world_point,
                            const CameraParameters& camera_params,
                            const Eigen::Matrix4d& Tcb,
                            const Eigen::Matrix2d& information)
    : m_observation(observation), m_world_point(world_point), 
      m_camera_params(camera_params), m_Tcb(Tcb), m_information(information), m_is_outlier(false) {}

bool PnPFactor::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
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

    // Extract SE3 parameters from tangent space (Ceres order: tx,ty,tz,rx,ry,rz)
    Eigen::Map<const Eigen::Vector6d> se3_tangent(parameters[0]);
    
    // Convert tangent space to SE3 using Sophus exp (consistent with parameterization)
    Sophus::SE3d T_wb = Sophus::SE3d::exp(se3_tangent);
    
    // Convert T_wb to T_cw transformation
    Eigen::Matrix3d R_wb = T_wb.rotationMatrix();
    Eigen::Vector3d t_wb = T_wb.translation();
    Eigen::Matrix3d R_bw = R_wb.transpose();
    Eigen::Vector3d t_bw = -R_bw * t_wb;
    
    // T_cw = T_cb * T_bw
    Eigen::Matrix3d R_cw = m_Tcb.block<3, 3>(0, 0) * R_bw;
    Eigen::Vector3d t_cw = m_Tcb.block<3, 3>(0, 0) * t_bw + m_Tcb.block<3, 1>(0, 3);
    
    // Transform world point to camera coordinates: Pc = R_cw * Pw + t_cw
    Eigen::Vector3d point_camera = R_cw * m_world_point + t_cw;
    
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
        
        // Jacobian calculation
        // Camera extrinsic rotation matrix
        Eigen::Matrix3d Rcb = m_Tcb.block<3, 3>(0, 0);
        
        // Body frame coordinates: Pb = Rbw * Pw + tbw
        Eigen::Vector3d Pb = R_bw * m_world_point + t_bw;
        
        // Jacobian of projection error w.r.t. camera coordinates: ∂(error)/∂(Pc)
        Eigen::Matrix<double, 2, 3> J_error_wrt_Pc;
        J_error_wrt_Pc(0, 0) = -m_camera_params.fx * z_inv;
        J_error_wrt_Pc(0, 1) = 0.0;
        J_error_wrt_Pc(0, 2) = x * m_camera_params.fx * z_inv_sq;
        J_error_wrt_Pc(1, 0) = 0.0;
        J_error_wrt_Pc(1, 1) = -m_camera_params.fy * z_inv;
        J_error_wrt_Pc(1, 2) = y * m_camera_params.fy * z_inv_sq;
        
        // Chain rule: ∂(error)/∂(twist) = ∂(error)/∂(Pc) * ∂(Pc)/∂(twist)
        
        // Translation part: ∂(Pc)/∂(translation) = -Rcb
        Eigen::Matrix<double, 2, 3> J_translation = J_error_wrt_Pc * (-Rcb);
        
        // Rotation part: ∂(Pc)/∂(rotation) = Rcb * [Pb]× for right perturbation
        Eigen::Matrix3d hatPb = Sophus::SO3d::hat(Pb);
        Eigen::Matrix<double, 2, 3> J_rotation = J_error_wrt_Pc * Rcb * hatPb;
        
        // Combine Jacobian components [translation | rotation] for Ceres order
        Eigen::Matrix<double, 2, 6> unweighted_jac;
        unweighted_jac << J_translation, J_rotation;
        
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

double PnPFactor::compute_chi_square(double const* const* parameters) const {
    // Extract SE3 parameters from tangent space (Ceres order: tx,ty,tz,rx,ry,rz)
    Eigen::Map<const Eigen::Vector6d> se3_tangent(parameters[0]);
    
    // Convert tangent space to SE3 using Sophus exp (consistent with parameterization)
    Sophus::SE3d T_wb = Sophus::SE3d::exp(se3_tangent);
    
    // Convert T_wb to T_cw transformation
    Eigen::Matrix3d R_wb = T_wb.rotationMatrix();
    Eigen::Vector3d t_wb = T_wb.translation();
    Eigen::Matrix3d R_bw = R_wb.transpose();
    Eigen::Vector3d t_bw = -R_bw * t_wb;
    
    // T_cw = T_cb * T_bw
    Eigen::Matrix3d R_cw = m_Tcb.block<3, 3>(0, 0) * R_bw;
    Eigen::Vector3d t_cw = m_Tcb.block<3, 3>(0, 0) * t_bw + m_Tcb.block<3, 1>(0, 3);
    
    // Transform world point to camera coordinates: Pc = R_cw * Pw + t_cw
    Eigen::Vector3d point_camera = R_cw * m_world_point + t_cw;
    
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
