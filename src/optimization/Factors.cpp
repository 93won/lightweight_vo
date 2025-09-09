/**
 * @file      Factors.cpp
 * @brief     Implements Ceres cost functions (factors) for VIO optimization.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-08-28
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "optimization/Factors.h"
#include "processing/IMUHandler.h"  // For IMUPreintegration
#include <limits>
#include <spdlog/spdlog.h>
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
        
        // Apply information matrix weighting using Cholesky decomposition: r_weighted = sqrt(Info) * r
        Eigen::LLT<Eigen::Matrix2d> llt(m_information);
        if (llt.info() == Eigen::Success) {
            // Use Cholesky decomposition: Information = L * L^T
            // Weighted residual = L * residual
            Eigen::Vector2d weighted_error = llt.matrixL() * error;
            residuals[0] = weighted_error[0];
            residuals[1] = weighted_error[1];
        } else {
            // Fallback to unweighted if Cholesky fails
            residuals[0] = error[0];
            residuals[1] = error[1];
        }
        
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
            
            // Apply information matrix weighting to projection jacobian using Cholesky
            Eigen::LLT<Eigen::Matrix2d> llt(m_information);
            Eigen::Matrix<double, 2, 3> weighted_J_proj;
            if (llt.info() == Eigen::Success) {
                weighted_J_proj = llt.matrixL() * J_proj_camera;
            } else {
                weighted_J_proj = J_proj_camera;
            }
            
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
    
    // Compute residual (which is already weighted by information matrix in Evaluate())
    double residuals[2];
    Evaluate(parameters, residuals, nullptr);
    
    // Chi-square error: weighted_residual^T * weighted_residual
    // Since Evaluate() returns Information * error, this gives us error^T * Information^T * Information * error
    // For symmetric positive definite Information matrix: Information^T = Information
    // So this becomes: error^T * Information^2 * error, which is incorrect for chi-square test
    
    // We need to compute the unweighted error and then apply information matrix properly
    // Let's recompute the unweighted error for proper chi-square calculation
    
    // Extract SE3 pose from parameters
    Eigen::Map<const Eigen::Vector6d> se3_tangent(parameters[0]);
    Sophus::SE3d Twb = Sophus::SE3d::exp(se3_tangent);
    
    // Extract 3D point
    Eigen::Map<const Eigen::Vector3d> world_point(parameters[1]);
    
    // Transform world point to camera coordinates
    Eigen::Matrix3d Rwb = Twb.rotationMatrix();
    Eigen::Vector3d twb = Twb.translation();
    Eigen::Matrix3d Rbw = Rwb.transpose();
    
    // Body coordinates: Pb = Rbw * (Pw - twb)
    Eigen::Vector3d body_point = Rbw * (world_point - twb);
    
    // Camera coordinates: Pc = Tcb * [Pb; 1]
    Eigen::Vector4d body_point_h(body_point.x(), body_point.y(), body_point.z(), 1.0);
    Eigen::Vector4d camera_point_h = m_Tcb * body_point_h;
    Eigen::Vector3d camera_point = camera_point_h.head<3>();
    
    // Check if point is in front of camera
    if (camera_point.z() <= 0.0) {
        return 1000.0; // Large chi-square for points behind camera
    }
    
    // Project to pixel coordinates
    double invz = 1.0 / camera_point.z();
    double u = m_camera_params.fx * camera_point.x() * invz + m_camera_params.cx;
    double v = m_camera_params.fy * camera_point.y() * invz + m_camera_params.cy;
    
    // Compute unweighted error
    Eigen::Vector2d projected(u, v);
    Eigen::Vector2d error = m_observation - projected;
    
    // Proper chi-square: error^T * Information * error
    double chi_square = error.transpose() * m_information * error;
    
    return chi_square;
}

// ===============================================================================
// IMU FACTOR IMPLEMENTATION
// ===============================================================================

bool IMUFactor::Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    
    // ===============================================================================
    // STEP 1: Extract parameters from optimization variables
    // ===============================================================================
    
    // pose1: [ρ1, φ1] -> [t1, R1]
    Eigen::Map<const Eigen::Vector3d> rho1(parameters[0]);     // translation
    Eigen::Map<const Eigen::Vector3d> phi1(parameters[0] + 3); // rotation (axis-angle)
    
    // Check for numerical stability before creating SO3
    Eigen::Vector3d phi1_clamped = phi1;
    double phi1_norm = phi1_clamped.norm();
    if (phi1_norm > M_PI) {
        phi1_clamped = phi1_clamped * (M_PI / phi1_norm);
    }
    
    Sophus::SO3d SO3_1 = Sophus::SO3d::exp(phi1_clamped);
    Eigen::Matrix3d R1 = SO3_1.matrix();
    Eigen::Vector3d t1 = rho1;
    
    // speed_bias1: [v1, ba1, bg1]
    Eigen::Map<const Eigen::Vector3d> v1(parameters[1]);        // velocity
    Eigen::Map<const Eigen::Vector3d> ba1(parameters[1] + 3);   // accel bias
    Eigen::Map<const Eigen::Vector3d> bg1(parameters[1] + 6);   // gyro bias
    
    // pose2: [ρ2, φ2] -> [t2, R2]  
    Eigen::Map<const Eigen::Vector3d> rho2(parameters[2]);
    Eigen::Map<const Eigen::Vector3d> phi2(parameters[2] + 3);
    
    // Check for numerical stability before creating SO3
    Eigen::Vector3d phi2_clamped = phi2;
    double phi2_norm = phi2_clamped.norm();
    if (phi2_norm > M_PI) {
        phi2_clamped = phi2_clamped * (M_PI / phi2_norm);
    }
    
    Sophus::SO3d SO3_2 = Sophus::SO3d::exp(phi2_clamped);
    Eigen::Matrix3d R2 = SO3_2.matrix();
    Eigen::Vector3d t2 = rho2;
    
    // speed_bias2: [v2, ba2, bg2]
    Eigen::Map<const Eigen::Vector3d> v2(parameters[3]);
    Eigen::Map<const Eigen::Vector3d> ba2(parameters[3] + 3);
    Eigen::Map<const Eigen::Vector3d> bg2(parameters[3] + 6);
    
    // ===============================================================================
    // STEP 2: Compute IMU predictions using preintegration
    // ===============================================================================
    
    double dt = m_preintegration->dt_total;
    
    // Cast float matrices to double for computation
    Eigen::Matrix3d delta_R = m_preintegration->delta_R.cast<double>();
    Eigen::Vector3d delta_V = m_preintegration->delta_V.cast<double>();
    Eigen::Vector3d delta_P = m_preintegration->delta_P.cast<double>();
    
    // Bias corrections (using precomputed Jacobians)
    Eigen::Vector3d delta_bg = bg1 - m_preintegration->gyro_bias.cast<double>();
    Eigen::Vector3d delta_ba = ba1 - m_preintegration->accel_bias.cast<double>();
    
    Eigen::Matrix3d JRg = m_preintegration->JRg.cast<double>();
    Eigen::Matrix3d JVg = m_preintegration->JVg.cast<double>();
    Eigen::Matrix3d JVa = m_preintegration->JVa.cast<double>();
    Eigen::Matrix3d JPg = m_preintegration->JPg.cast<double>();
    Eigen::Matrix3d JPa = m_preintegration->JPa.cast<double>();
    
    // Corrected preintegration with bias updates
    Eigen::Vector3d bias_correction_rot = JRg * delta_bg;
    
    // Normalize rotation matrix using SVD to ensure orthogonality before correction
    Eigen::JacobiSVD<Eigen::Matrix3d> svd_init(delta_R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    delta_R = svd_init.matrixU() * svd_init.matrixV().transpose();
    
    Eigen::Matrix3d delta_R_corrected = delta_R * Sophus::SO3d::exp(bias_correction_rot).matrix();
    
    // Final normalization to ensure orthogonality
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(delta_R_corrected, Eigen::ComputeFullU | Eigen::ComputeFullV);
    delta_R_corrected = svd.matrixU() * svd.matrixV().transpose();
    
    Eigen::Vector3d delta_V_corrected = delta_V + JVg * delta_bg + JVa * delta_ba;
    Eigen::Vector3d delta_P_corrected = delta_P + JPg * delta_bg + JPa * delta_ba;
    
    // ===============================================================================  
    // STEP 3: Compute residuals
    // ===============================================================================
    
    Eigen::Map<Eigen::Vector3d> residual_p(residuals);      // position residual
    Eigen::Map<Eigen::Vector3d> residual_R(residuals + 3);  // rotation residual  
    Eigen::Map<Eigen::Vector3d> residual_v(residuals + 6);  // velocity residual
    Eigen::Map<Eigen::Vector3d> residual_ba(residuals + 9); // accel bias residual
    Eigen::Map<Eigen::Vector3d> residual_bg(residuals + 12);// gyro bias residual
    
    // Position residual: r_p = (t2 - t1) - (R1 * δP + v1*dt + 0.5*g*dt²)
    residual_p = (t2 - t1) - (R1 * delta_P_corrected + v1 * dt + 0.5 * m_gravity * dt * dt);
    
    // Rotation residual: r_R = Log(δR⁻¹ * R1⁻¹ * R2)
    Eigen::Matrix3d rotation_error = delta_R_corrected.transpose() * R1.transpose() * R2;
    residual_R = log_SO3(rotation_error);
    
    // Velocity residual: r_v = (v2 - v1) - (R1 * δV + g*dt)  
    residual_v = (v2 - v1) - (R1 * delta_V_corrected + m_gravity * dt);
    
    // Bias residuals (random walk model): Δbias ~ N(0, σ²*dt)
    residual_ba = ba2 - ba1;
    residual_bg = bg2 - bg1;
    
    // ===============================================================================
    // STEP 4: Compute Jacobians (if requested)
    // ===============================================================================
    
    if (jacobians != nullptr) {
        
        // Jacobian w.r.t. pose1 [ρ1, φ1] (6x15)
        if (jacobians[0] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> J1(jacobians[0]);
            J1.setZero();
            
            // ∂r_p/∂ρ1 = -I
            J1.block<3,3>(0, 0) = -Eigen::Matrix3d::Identity();
            
            // ∂r_p/∂φ1 = -[R1*δP]_× (skew-symmetric of R1*δP)
            J1.block<3,3>(0, 3) = -skew_symmetric(R1 * delta_P_corrected);
            
            // ∂r_R/∂φ1 = -Jr⁻¹(r_R) * (δR)ᵀ
            Eigen::Matrix3d Jr_inv = left_jacobian_SO3(-residual_R);
            J1.block<3,3>(3, 3) = -Jr_inv * delta_R_corrected.transpose();
            
            // ∂r_v/∂φ1 = -[R1*δV]_×
            J1.block<3,3>(6, 3) = -skew_symmetric(R1 * delta_V_corrected);
        }
        
        // Jacobian w.r.t. speed_bias1 [v1, ba1, bg1] (9x15)
        if (jacobians[1] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J2(jacobians[1]);
            J2.setZero();
            
            // ∂r_p/∂v1 = -dt*I
            J2.block<3,3>(0, 0) = -dt * Eigen::Matrix3d::Identity();
            
            // ∂r_p/∂ba1 = -R1 * JPa
            J2.block<3,3>(0, 3) = -R1 * JPa;
            
            // ∂r_p/∂bg1 = -R1 * JPg  
            J2.block<3,3>(0, 6) = -R1 * JPg;
            
            // ∂r_R/∂bg1 = -Jr⁻¹(r_R) * δRᵀ * Jr(JRg*Δbg) * JRg
            Eigen::Matrix3d Jr_inv = left_jacobian_SO3(-residual_R);
            Eigen::Matrix3d Jr_bias = right_jacobian_SO3(JRg * delta_bg);
            J2.block<3,3>(3, 6) = -Jr_inv * delta_R_corrected.transpose() * Jr_bias * JRg;
            
            // ∂r_v/∂v1 = -I
            J2.block<3,3>(6, 0) = -Eigen::Matrix3d::Identity();
            
            // ∂r_v/∂ba1 = -R1 * JVa
            J2.block<3,3>(6, 3) = -R1 * JVa;
            
            // ∂r_v/∂bg1 = -R1 * JVg
            J2.block<3,3>(6, 6) = -R1 * JVg;
            
            // ∂r_ba/∂ba1 = -I
            J2.block<3,3>(9, 3) = -Eigen::Matrix3d::Identity();
            
            // ∂r_bg/∂bg1 = -I  
            J2.block<3,3>(12, 6) = -Eigen::Matrix3d::Identity();
        }
        
        // Jacobian w.r.t. pose2 [ρ2, φ2] (6x15)
        if (jacobians[2] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> J3(jacobians[2]);
            J3.setZero();
            
            // ∂r_p/∂ρ2 = I
            J3.block<3,3>(0, 0) = Eigen::Matrix3d::Identity();
            
            // ∂r_R/∂φ2 = Jr⁻¹(r_R)
            Eigen::Matrix3d Jr_inv = left_jacobian_SO3(-residual_R);
            J3.block<3,3>(3, 3) = Jr_inv;
        }
        
        // Jacobian w.r.t. speed_bias2 [v2, ba2, bg2] (9x15)
        if (jacobians[3] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J4(jacobians[3]);
            J4.setZero();
            
            // ∂r_v/∂v2 = I
            J4.block<3,3>(6, 0) = Eigen::Matrix3d::Identity();
            
            // ∂r_ba/∂ba2 = I
            J4.block<3,3>(9, 3) = Eigen::Matrix3d::Identity();
            
            // ∂r_bg/∂bg2 = I
            J4.block<3,3>(12, 6) = Eigen::Matrix3d::Identity();
        }
    }
    
    return true;
}

Eigen::Matrix3d IMUFactor::skew_symmetric(const Eigen::Vector3d& v) const {
    Eigen::Matrix3d skew = Eigen::Matrix3d::Zero();
    skew(0, 1) = -v(2);
    skew(0, 2) =  v(1);
    skew(1, 0) =  v(2);
    skew(1, 2) = -v(0);
    skew(2, 0) = -v(1);
    skew(2, 1) =  v(0);
    return skew;
}

Eigen::Matrix3d IMUFactor::right_jacobian_SO3(const Eigen::Vector3d& phi) const {
    double theta = phi.norm();
    if (theta < 1e-6) {
        return Eigen::Matrix3d::Identity() - 0.5 * skew_symmetric(phi);
    }
    
    Eigen::Vector3d axis = phi / theta;
    double c = cos(theta);
    double s = sin(theta);
    
    return s / theta * Eigen::Matrix3d::Identity() + 
           (1.0 - c) / theta * skew_symmetric(axis) + 
           (theta - s) / theta * axis * axis.transpose();
}

Eigen::Matrix3d IMUFactor::left_jacobian_SO3(const Eigen::Vector3d& phi) const {
    return right_jacobian_SO3(-phi).transpose();
}

Eigen::Vector3d IMUFactor::log_SO3(const Eigen::Matrix3d& R) const {
    // Normalize rotation matrix using SVD to ensure orthogonality before log operation
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d R_normalized = svd.matrixU() * svd.matrixV().transpose();
    
    return Sophus::SO3d(R_normalized).log();
}

Eigen::Matrix4d IMUFactor::tangent_to_matrix(const Sophus::SE3d::Tangent& tangent) const {
    return Sophus::SE3d::exp(tangent).matrix();
}

// ===============================================================================
// INERTIAL GRAVITY FACTOR IMPLEMENTATION
// ===============================================================================

bool InertialGravityFactor::Evaluate(double const* const* parameters,
                                    double* residuals,
                                    double** jacobians) const {
    
    // ===============================================================================
    // STEP 1: Extract parameters from optimization variables
    // ===============================================================================
    
    // posei: SE3 pose [ti, Ri] (Twb format)
    Eigen::Map<const Eigen::Vector6d> posei_tangent(parameters[0]);
    Sophus::SE3d T_wbi = Sophus::SE3d::exp(posei_tangent);
    Eigen::Matrix3d R_wbi = T_wbi.rotationMatrix();
    Eigen::Vector3d t_wbi = T_wbi.translation();
    Eigen::Matrix3d R_bwi = R_wbi.transpose();
    
    // velocity_biasi: [vi, bai, bgi]
    Eigen::Map<const Eigen::Vector3d> vi(parameters[1]);        // velocity
    Eigen::Map<const Eigen::Vector3d> bai(parameters[1] + 3);   // accel bias
    Eigen::Map<const Eigen::Vector3d> bgi(parameters[1] + 6);   // gyro bias
    
    // posej: SE3 pose [tj, Rj] (Twb format)
    Eigen::Map<const Eigen::Vector6d> posej_tangent(parameters[2]);
    Sophus::SE3d T_wbj = Sophus::SE3d::exp(posej_tangent);
    Eigen::Matrix3d R_wbj = T_wbj.rotationMatrix();
    Eigen::Vector3d t_wbj = T_wbj.translation();
    
    // velocity_biasj: [vj, baj, bgj] 
    Eigen::Map<const Eigen::Vector3d> vj(parameters[3]);
    Eigen::Map<const Eigen::Vector3d> baj(parameters[3] + 3);
    Eigen::Map<const Eigen::Vector3d> bgj(parameters[3] + 6);
    
    // gravity_dir: 2D gravity direction parameterization
    Eigen::Map<const Eigen::Vector2d> gravity_dir(parameters[4]);
    
    // ===============================================================================
    // STEP 2: Compute gravity vector from direction parameterization
    // ===============================================================================
    
    // Convert 2D gravity direction to full 3D gravity vector
    Eigen::Matrix3d R_wg = gravity_dir_to_rotation(gravity_dir);
    Eigen::Vector3d g_I(0, 0, -m_gravity_magnitude);  // gravity in gravity frame
    Eigen::Vector3d g = R_wg * g_I;  // gravity in world frame

    // ===============================================================================
    // STEP 3: Compute IMU predictions using preintegration with bias correction
    // ===============================================================================
    
    double dt = m_preintegration->dt_total;
    
    // Cast float matrices to double for computation
    Eigen::Matrix3d delta_R = m_preintegration->delta_R.cast<double>();
    Eigen::Vector3d delta_V = m_preintegration->delta_V.cast<double>();
    Eigen::Vector3d delta_P = m_preintegration->delta_P.cast<double>();
    
    // Bias corrections (using precomputed Jacobians)
    Eigen::Vector3d delta_bg = bgi - m_preintegration->gyro_bias.cast<double>();
    Eigen::Vector3d delta_ba = bai - m_preintegration->accel_bias.cast<double>();
    
    Eigen::Matrix3d JRg = m_preintegration->JRg.cast<double>();
    Eigen::Matrix3d JVg = m_preintegration->JVg.cast<double>();
    Eigen::Matrix3d JVa = m_preintegration->JVa.cast<double>();
    Eigen::Matrix3d JPg = m_preintegration->JPg.cast<double>();
    Eigen::Matrix3d JPa = m_preintegration->JPa.cast<double>();
    
    // Corrected preintegration with bias updates
    Eigen::Vector3d bias_correction_rot = JRg * delta_bg;
    
    // Normalize rotation matrix using SVD to ensure orthogonality before correction
    Eigen::JacobiSVD<Eigen::Matrix3d> svd_init(delta_R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    delta_R = svd_init.matrixU() * svd_init.matrixV().transpose();
    
    Eigen::Matrix3d delta_R_corrected = delta_R * Sophus::SO3d::exp(bias_correction_rot).matrix();
    
    // Final normalization to ensure orthogonality
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(delta_R_corrected, Eigen::ComputeFullU | Eigen::ComputeFullV);
    delta_R_corrected = svd.matrixU() * svd.matrixV().transpose();
    
    Eigen::Vector3d delta_V_corrected = delta_V + JVg * delta_bg + JVa * delta_ba;
    Eigen::Vector3d delta_P_corrected = delta_P + JPg * delta_bg + JPa * delta_ba;
    
    // ===============================================================================  
    // STEP 4: Compute residuals in Body Frame
    // ===============================================================================
    
    Eigen::Map<Eigen::Vector3d> residual_R(residuals);      // rotation residual  
    Eigen::Map<Eigen::Vector3d> residual_v(residuals + 3);  // velocity residual
    Eigen::Map<Eigen::Vector3d> residual_p(residuals + 6);  // position residual
    
    // Rotation residual: r_R = Log(δR^T * Ri^T * Rj) 
    Eigen::Matrix3d rotation_error = delta_R_corrected.transpose() * R_bwi * R_wbj;
    residual_R = log_SO3(rotation_error);
    
    // Velocity residual: r_v = Ri^T * ((vj - vi) - g*dt) - δV
    residual_v = R_bwi * ((vj - vi) - g * dt) - delta_V_corrected;

    // Position residual: r_p = Ri^T * ((tj - ti - vi*dt) - g*dt²/2) - δP
    residual_p = R_bwi * ((t_wbj - t_wbi - vi * dt) - 0.5 * g * dt * dt) - delta_P_corrected;
    
    // ===============================================================================
    // STEP 5: Compute Jacobians (if requested)
    // ===============================================================================
    
    if (jacobians != nullptr) {
        
        // Precompute some commonly used terms
        Eigen::Matrix3d Jr_inv = left_jacobian_SO3(-residual_R);
        Eigen::Matrix3d eR = delta_R_corrected.transpose() * R_bwi * R_wbj;
        
        // Gravity direction Jacobian matrix: dG/dTheta
        Eigen::MatrixXd Gm = Eigen::MatrixXd::Zero(3, 2);
        Gm(0, 1) = -m_gravity_magnitude;  // dG/dtheta_y component in x direction
        Gm(1, 0) = m_gravity_magnitude;   // dG/dtheta_x component in y direction
        Eigen::MatrixXd dGdTheta = R_wg * Gm;
        
        // Jacobian w.r.t. posei [ti, Ri] - NOTE: Our order is [translation, rotation]
        if (jacobians[0] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 6, Eigen::RowMajor>> Ji(jacobians[0]);
            Ji.setZero();
            
            // ∂r_R/∂Ri = -Jr^(-1) * Rj^T * Ri (rotation part, indices 3-5)
            Ji.block<3,3>(0, 3) = -Jr_inv * R_wbj.transpose() * R_wbi;
            
            // ∂r_v/∂Ri = [Ri^T * ((vj - vi) - g*dt)]_× (rotation part, indices 3-5)
            Ji.block<3,3>(3, 3) = skew_symmetric(R_bwi * ((vj - vi) - g * dt));
            
            // ∂r_p/∂ti = -Ri^T (translation part, indices 0-2)
            Ji.block<3,3>(6, 0) = -R_bwi;
            
            // ∂r_p/∂Ri = [Ri^T * ((tj - ti - vi*dt) - g*dt²/2)]_× (rotation part, indices 3-5)
            Ji.block<3,3>(6, 3) = skew_symmetric(R_bwi * ((t_wbj - t_wbi - vi * dt) - 0.5 * g * dt * dt));
        }
        
        // Jacobian w.r.t. velocity_biasi [vi, bai, bgi]
        if (jacobians[1] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor>> Jvi(jacobians[1]);
            Jvi.setZero();
            
            // ∂r_R/∂bgi = -Jr^(-1) * eR^T * Jr(JRg*δbg) * JRg
            Eigen::Matrix3d Jr_bias = right_jacobian_SO3(JRg * delta_bg);
            Jvi.block<3,3>(0, 6) = -Jr_inv * eR.transpose() * Jr_bias * JRg;
            
            // ∂r_v/∂vi = -Ri^T
            Jvi.block<3,3>(3, 0) = -R_bwi;
            
            // ∂r_v/∂bai = -JVa
            Jvi.block<3,3>(3, 3) = -JVa;
            
            // ∂r_v/∂bgi = -JVg
            Jvi.block<3,3>(3, 6) = -JVg;
            
            // ∂r_p/∂vi = -Ri^T * dt
            Jvi.block<3,3>(6, 0) = -R_bwi * dt;
            
            // ∂r_p/∂bai = -JPa
            Jvi.block<3,3>(6, 3) = -JPa;
            
            // ∂r_p/∂bgi = -JPg
            Jvi.block<3,3>(6, 6) = -JPg;
        }
        
        // Jacobian w.r.t. posej [tj, Rj] - NOTE: Our order is [translation, rotation]
        if (jacobians[2] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 6, Eigen::RowMajor>> Jj(jacobians[2]);
            Jj.setZero();
            
            // ∂r_R/∂Rj = Jr^(-1) (rotation part, indices 3-5)
            Jj.block<3,3>(0, 3) = Jr_inv;
            
            // ∂r_p/∂tj = Ri^T (translation part, indices 0-2)
            Jj.block<3,3>(6, 0) = R_bwi;
        }
        
        // Jacobian w.r.t. velocity_biasj [vj, baj, bgj]
        if (jacobians[3] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor>> Jvj(jacobians[3]);
            Jvj.setZero();
            
            // ∂r_v/∂vj = Ri^T
            Jvj.block<3,3>(3, 0) = R_bwi;
        }
        
        // Jacobian w.r.t. gravity_dir [theta_x, theta_y]
        if (jacobians[4] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 2, Eigen::RowMajor>> Jg(jacobians[4]);
            Jg.setZero();
            
            // ∂r_v/∂gravity_dir = -Ri^T * dG/dTheta * dt
            Jg.block<3,2>(3, 0) = -R_bwi * dGdTheta * dt;
            
            // ∂r_p/∂gravity_dir = -0.5 * Ri^T * dG/dTheta * dt²
            Jg.block<3,2>(6, 0) = -0.5 * R_bwi * dGdTheta * dt * dt;
        }
    }
    
    return true;
}

Eigen::Matrix3d InertialGravityFactor::skew_symmetric(const Eigen::Vector3d& v) const {
    Eigen::Matrix3d skew = Eigen::Matrix3d::Zero();
    skew(0, 1) = -v(2);
    skew(0, 2) =  v(1);
    skew(1, 0) =  v(2);
    skew(1, 2) = -v(0);
    skew(2, 0) = -v(1);
    skew(2, 1) =  v(0);
    return skew;
}

Eigen::Matrix3d InertialGravityFactor::right_jacobian_SO3(const Eigen::Vector3d& phi) const {
    double theta = phi.norm();
    if (theta < 1e-6) {
        return Eigen::Matrix3d::Identity() - 0.5 * skew_symmetric(phi);
    }
    
    Eigen::Vector3d axis = phi / theta;
    double c = cos(theta);
    double s = sin(theta);
    
    return s / theta * Eigen::Matrix3d::Identity() + 
           (1.0 - c) / theta * skew_symmetric(axis) + 
           (theta - s) / theta * axis * axis.transpose();
}

Eigen::Matrix3d InertialGravityFactor::left_jacobian_SO3(const Eigen::Vector3d& phi) const {
    return right_jacobian_SO3(-phi).transpose();
}

Eigen::Vector3d InertialGravityFactor::log_SO3(const Eigen::Matrix3d& R) const {
    // Normalize rotation matrix using SVD to ensure orthogonality before log operation
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d R_normalized = svd.matrixU() * svd.matrixV().transpose();
    
    return Sophus::SO3d(R_normalized).log();
}

Eigen::Matrix3d InertialGravityFactor::gravity_dir_to_rotation(const Eigen::Vector2d& gravity_dir) const {
    // Convert 2D gravity direction parameterization to rotation matrix
    // Rotate around x and y axes only
    double theta_x = gravity_dir[0];
    double theta_y = gravity_dir[1];
    
    // Create rotation matrix: R = Ry(theta_y) * Rx(theta_x)
    Eigen::Matrix3d R_x = Eigen::AngleAxisd(theta_x, Eigen::Vector3d::UnitX()).toRotationMatrix();
    Eigen::Matrix3d R_y = Eigen::AngleAxisd(theta_y, Eigen::Vector3d::UnitY()).toRotationMatrix();
    
    return R_y * R_x;
}

} // namespace factor
} // namespace lightweight_vio
