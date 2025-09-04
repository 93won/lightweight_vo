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

// BAFactor implementation
BAFactor::BAFactor(const Eigen::Vector2d& observation,
                   const CameraParameters& camera_params,
                   const Eigen::Matrix4d& Tcb,
                   const Eigen::Matrix2d& information)
    : m_observation(observation)
    , m_camera_params(camera_params)
    , m_Tcb(Tcb)
    , m_information(information)
    , m_is_outlier(false) {
}

bool BAFactor::Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    
    if (m_is_outlier) {
        // If marked as outlier, set zero residual and jacobians
        residuals[0] = 0.0;
        residuals[1] = 0.0;
        
        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jac_pose(jacobians[0]);
                jac_pose.setZero();
            }
            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jac_point(jacobians[1]);
                jac_point.setZero();
            }
        }
        return true;
    }
    
    try {
        // Extract pose parameters (SE3 tangent space)
        Eigen::Map<const Eigen::Vector6d> pose_tangent(parameters[0]);
        
        // Extract 3D point in world coordinates
        Eigen::Map<const Eigen::Vector3d> world_point(parameters[1]);
        
        // Convert tangent space to SE3 (Twb: body to world)
        Sophus::SE3d Twb = Sophus::SE3d::exp(pose_tangent);
        
        // Get rotation and translation matrices
        Eigen::Matrix3d Rwb = Twb.rotationMatrix();
        Eigen::Vector3d twb = Twb.translation();
        Eigen::Matrix3d Rbw = Rwb.transpose();
        Eigen::Vector3d tbw = -Rbw * twb;
        
        // Compute Tcw (camera to world transformation)
        // Tcw = Tcb * Tbw = Tcb * Twb^(-1)
        Eigen::Matrix3d Rcw = m_Tcb.block<3, 3>(0, 0) * Rbw;
        Eigen::Vector3d tcw = m_Tcb.block<3, 3>(0, 0) * tbw + m_Tcb.block<3, 1>(0, 3);
        
        // Transform world point to camera coordinates
        Eigen::Vector3d camera_point = Rcw * world_point + tcw;
        
        double x = camera_point[0];
        double y = camera_point[1];
        double z = camera_point[2];
        double invz = 1.0 / (z + 1e-9);
        
        // Check for valid depth
        if (invz < 1e-3 || invz > 1e2) {
            // Behind camera or too far - return large residual
            residuals[0] = 640.0;
            residuals[1] = 360.0;
            return true;
        }
        
        // Project to pixel coordinates
        double u = m_camera_params.fx * x * invz + m_camera_params.cx;
        double v = m_camera_params.fy * y * invz + m_camera_params.cy;
        
        // Compute residual: observed - projected
        Eigen::Vector2d projected(u, v);
        Eigen::Vector2d error = m_observation - projected;
        
        // Apply information matrix (precision matrix)
        Eigen::Vector2d weighted_error = m_information * error;
        
        residuals[0] = weighted_error[0];
        residuals[1] = weighted_error[1];
        
        // Compute Jacobians if requested
        if (jacobians) {
            // Camera projection Jacobian w.r.t camera point
            Eigen::Matrix<double, 2, 3> J_proj_camera;
            J_proj_camera(0, 0) = -m_camera_params.fx * invz;
            J_proj_camera(0, 1) = 0.0;
            J_proj_camera(0, 2) = x * m_camera_params.fx * invz * invz;
            J_proj_camera(1, 0) = 0.0;
            J_proj_camera(1, 1) = -m_camera_params.fy * invz;
            J_proj_camera(1, 2) = y * m_camera_params.fy * invz * invz;
            
            // Apply information matrix to projection jacobian
            Eigen::Matrix<double, 2, 3> weighted_J_proj = m_information * J_proj_camera;
            
            // Jacobian w.r.t pose (SE3 tangent space)
            if (jacobians[0]) {
                // Body point in body frame: Pb = Rbw * (Pw - twb)
                Eigen::Vector3d body_point = Rbw * (world_point - twb);
                
                // Jacobian of camera point w.r.t body translation
                Eigen::Matrix<double, 3, 3> J_camera_trans = -m_Tcb.block<3, 3>(0, 0);
                
                // Jacobian of camera point w.r.t body rotation  
                Eigen::Matrix<double, 3, 3> J_camera_rot = m_Tcb.block<3, 3>(0, 0) * Sophus::SO3d::hat(body_point);
                
                // Combine rotation and translation jacobians [3x6]
                Eigen::Matrix<double, 3, 6> J_camera_pose;
                J_camera_pose.block<3, 3>(0, 0) = J_camera_rot;  // w.r.t rotation
                J_camera_pose.block<3, 3>(0, 3) = J_camera_trans; // w.r.t translation
                
                // Chain rule: J_residual_pose = J_proj_camera * J_camera_pose
                Eigen::Matrix<double, 2, 6> J_pose = weighted_J_proj * J_camera_pose;
                
                Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jac_pose(jacobians[0]);
                jac_pose = J_pose;
            }
            
            // Jacobian w.r.t 3D point
            if (jacobians[1]) {
                // Jacobian of camera point w.r.t world point
                Eigen::Matrix<double, 3, 3> J_camera_point = m_Tcb.block<3, 3>(0, 0) * Rbw;
                
                // Chain rule: J_residual_point = J_proj_camera * J_camera_point  
                Eigen::Matrix<double, 2, 3> J_point = weighted_J_proj * J_camera_point;
                
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jac_point(jacobians[1]);
                jac_point = J_point;
            }
        }
        
        return true;
        
    } catch (const std::exception& e) {
        return false;
    }
}

double BAFactor::compute_chi_square(double const* const* parameters) const {
    if (m_is_outlier) {
        return 0.0;
    }
    
    // Compute residual
    double residuals[2];
    Evaluate(parameters, residuals, nullptr);
    
    // Chi-square error: residual^T * Information * residual
    Eigen::Vector2d res(residuals[0], residuals[1]);
    double chi_square = res.transpose() * res; // Already weighted by information matrix
    
    return chi_square;
}

} // namespace factor
} // namespace lightweight_vio
