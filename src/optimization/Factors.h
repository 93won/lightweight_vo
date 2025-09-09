/**
 * @file      Factors.h
 * @brief     Defines Ceres cost functions (factors) for VIO optimization.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-08-28
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <vector>
#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/sized_cost_function.h>
#include <ceres/manifold.h>
#include <ceres/local_parameterization.h>
#include <sophus/se3.hpp>

// Forward declaration
namespace lightweight_vio {
    struct IMUPreintegration;
}

// Define Vector6d as it's not available in standard Eigen
namespace Eigen {
    typedef Matrix<double, 6, 1> Vector6d;
}

namespace lightweight_vio {
namespace factor {

/**
 * @brief Camera parameters structure
 */
struct CameraParameters {
    double fx, fy, cx, cy;
    
    CameraParameters(double fx_, double fy_, double cx_, double cy_)
        : fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}
};

/**
 * @brief Monocular PnP cost function with analytical Jacobian
 * Uses Twb (body-to-world) pose with T_CB (body-to-camera) extrinsics
 * Residual: observed_pixel - projected_pixel
 * Parameters: SE3 pose in tangent space [6] (Twb)
 * Residual dimension: [2]
 */
class PnPFactor : public ceres::SizedCostFunction<2, 6> {
public:
    /**
     * @brief Constructor
     * @param observation Observed 2D pixel coordinates [u, v]
     * @param world_point 3D point in world coordinates
     * @param camera_params Camera intrinsic parameters
     * @param Tcb Body-to-camera transformation matrix [4x4] (T_CB)
     * @param information Information matrix (precision matrix) [2x2]
     */
    PnPFactor(const Eigen::Vector2d& observation,
                  const Eigen::Vector3d& world_point,
                  const CameraParameters& camera_params,
                  const Eigen::Matrix4d& Tcb,
                  const Eigen::Matrix2d& information = Eigen::Matrix2d::Identity());

    /**
     * @brief Set outlier flag to disable optimization for this factor
     * @param is_outlier If true, this factor will not contribute to optimization
     */
    void set_outlier(bool is_outlier) { m_is_outlier = is_outlier; }
    
    /**
     * @brief Get outlier flag
     * @return true if this factor is marked as outlier
     */
    bool is_outlier() const { return m_is_outlier; }

    /**
     * @brief Evaluate residual and Jacobian
     * @param parameters SE3 pose parameters in tangent space [6]
     * @param residuals Output residual [2]
     * @param jacobians Output Jacobian matrices [2x6] if not nullptr
     * @return true if evaluation successful
     */
    virtual bool Evaluate(double const* const* parameters,
                         double* residuals,
                         double** jacobians) const override;

    /**
     * @brief Compute Chi-square error for outlier detection
     * @param parameters SE3 pose parameters in tangent space [6]
     * @return Chi-square error value
     */
    double compute_chi_square(double const* const* parameters) const;

private:
    Eigen::Vector2d m_observation;    // Observed pixel coordinates
    Eigen::Vector3d m_world_point;    // 3D world coordinates
    CameraParameters m_camera_params; // Camera intrinsics
    Eigen::Matrix4d m_Tcb;            // Body-to-camera transformation (T_CB)
    Eigen::Matrix2d m_information;    // Information matrix (precision matrix)
    bool m_is_outlier;                // Outlier flag to disable optimization
};

/**
 * @brief Bundle Adjustment monocular cost function with analytical Jacobian
 * Uses Twb (body-to-world) pose with T_CB (body-to-camera) extrinsics
 * Residual: observed_pixel - projected_pixel
 * Parameters: SE3 pose in tangent space [6] (Twb), 3D point position [3]
 * Residual dimension: [2]
 */
class BAFactor : public ceres::SizedCostFunction<2, 6, 3> {
public:
    /**
     * @brief Constructor
     * @param observation Observed 2D pixel coordinates [u, v]
     * @param camera_params Camera intrinsic parameters
     * @param Tcb Body-to-camera transformation matrix [4x4] (T_CB)
     * @param information Information matrix (precision matrix) [2x2]
     */
    BAFactor(const Eigen::Vector2d& observation,
             const CameraParameters& camera_params,
             const Eigen::Matrix4d& Tcb,
             const Eigen::Matrix2d& information = Eigen::Matrix2d::Identity());

    /**
     * @brief Set outlier flag to disable optimization for this factor
     * @param is_outlier If true, this factor will not contribute to optimization
     */
    void set_outlier(bool is_outlier) { m_is_outlier = is_outlier; }
    
    /**
     * @brief Get outlier flag
     * @return true if this factor is marked as outlier
     */
    bool is_outlier() const { return m_is_outlier; }

    /**
     * @brief Evaluate residual and Jacobian
     * @param parameters[0] SE3 pose parameters in tangent space [6] (Twb)
     * @param parameters[1] 3D point position in world coordinates [3]
     * @param residuals Output residual [2]
     * @param jacobians[0] Output Jacobian w.r.t pose [2x6] if not nullptr
     * @param jacobians[1] Output Jacobian w.r.t point [2x3] if not nullptr
     * @return true if evaluation successful
     */
    virtual bool Evaluate(double const* const* parameters,
                         double* residuals,
                         double** jacobians) const override;

    /**
     * @brief Compute Chi-square error for outlier detection
     * @param parameters[0] SE3 pose parameters in tangent space [6]
     * @param parameters[1] 3D point position in world coordinates [3]
     * @return Chi-square error value
     */
    double compute_chi_square(double const* const* parameters) const;

private:
    Eigen::Vector2d m_observation;    // Observed pixel coordinates
    CameraParameters m_camera_params; // Camera intrinsics
    Eigen::Matrix4d m_Tcb;            // Body-to-camera transformation (T_CB)
    Eigen::Matrix2d m_information;    // Information matrix (precision matrix)
    bool m_is_outlier;                // Outlier flag to disable optimization
};

// ===============================================================================
// INERTIAL OPTIMIZATION COST FUNCTIONS
// ===============================================================================

/**
 * @brief IMU preintegration factor for pose graph optimization
 * 
 * Residual dimension: 15 (position:3, rotation:3, velocity:3, accel_bias:3, gyro_bias:3)
 * Parameter blocks: 4
 *   - pose1: SE(3) tangent space [ρ1, φ1] (6D) 
 *   - speed_bias1: [v1, ba1, bg1] (9D)
 *   - pose2: SE(3) tangent space [ρ2, φ2] (6D)
 *   - speed_bias2: [v2, ba2, bg2] (9D)
 */
class IMUFactor : public ceres::SizedCostFunction<15, 6, 9, 6, 9> {
public:
    /**
     * @brief Constructor
     * @param preintegration IMU preintegration measurement
     * @param gravity Gravity vector in world frame
     */
    IMUFactor(std::shared_ptr<IMUPreintegration> preintegration,
              const Eigen::Vector3d& gravity)
        : m_preintegration(preintegration), m_gravity(gravity) {}

    /**
     * @brief Evaluate residuals and Jacobians
     */
    virtual bool Evaluate(double const* const* parameters,
                         double* residuals,
                         double** jacobians) const override;

private:
    std::shared_ptr<IMUPreintegration> m_preintegration;
    Eigen::Vector3d m_gravity;

    /**
     * @brief Convert SE(3) tangent vector to transformation matrix
     * @param tangent 6D tangent vector [ρ, φ] (translation, rotation)
     * @return 4x4 transformation matrix
     */
    Eigen::Matrix4d tangent_to_matrix(const Sophus::SE3d::Tangent& tangent) const;

    /**
     * @brief Skew-symmetric matrix
     */
    Eigen::Matrix3d skew_symmetric(const Eigen::Vector3d& v) const;

    /**
     * @brief Right Jacobian of SO(3)
     */
    Eigen::Matrix3d right_jacobian_SO3(const Eigen::Vector3d& phi) const;

    /**
     * @brief Left Jacobian of SO(3)  
     */
    Eigen::Matrix3d left_jacobian_SO3(const Eigen::Vector3d& phi) const;

    /**
     * @brief Logarithm map of SO(3) (rotation matrix to axis-angle)
     */
    Eigen::Vector3d log_SO3(const Eigen::Matrix3d& R) const;
};

/**
 * @brief Inertial factor with gravity direction optimization
 * 
 * This factor connects two poses with velocity and bias parameters, and optimizes
 * gravity direction using body frame residuals for better numerical stability.
 * 
 * Parameter blocks:
 * - pose1: SE3 pose (6 DoF) [t, rotation]
 * - velocity_bias1: [v1, ba1, bg1] (9 DoF)
 * - pose2: SE3 pose (6 DoF) [t, rotation]
 * - velocity_bias2: [v2, ba2, bg2] (9 DoF)
 * - gravity_dir: 2D parameterization of gravity direction
 */
class InertialGravityFactor : public ceres::SizedCostFunction<9, 6, 9, 6, 9, 2> {
public:
    /**
     * @brief Constructor
     * @param preintegration IMU preintegration data
     * @param gravity_magnitude Magnitude of gravity (default: 9.81)
     */
    InertialGravityFactor(std::shared_ptr<IMUPreintegration> preintegration,
                          double gravity_magnitude = 9.81)
        : m_preintegration(preintegration), m_gravity_magnitude(gravity_magnitude) {}

    /**
     * @brief Evaluate residual and Jacobians 
     * Residual: [rotation_error, velocity_error, position_error] (9D)
     * Body frame approach for better numerical stability
     */
    virtual bool Evaluate(double const* const* parameters,
                         double* residuals,
                         double** jacobians) const override;private:
    std::shared_ptr<IMUPreintegration> m_preintegration;
    double m_gravity_magnitude;

    /**
     * @brief Skew-symmetric matrix
     */
    Eigen::Matrix3d skew_symmetric(const Eigen::Vector3d& v) const;

    /**
     * @brief Right Jacobian of SO(3)
     */
    Eigen::Matrix3d right_jacobian_SO3(const Eigen::Vector3d& phi) const;

    /**
     * @brief Left Jacobian of SO(3)  
     */
    Eigen::Matrix3d left_jacobian_SO3(const Eigen::Vector3d& phi) const;

    /**
     * @brief Logarithm map of SO(3) (rotation matrix to axis-angle)
     */
    Eigen::Vector3d log_SO3(const Eigen::Matrix3d& R) const;

    /**
     * @brief Convert 2D gravity direction to rotation matrix
     * @param gravity_dir 2D parameterization [theta_x, theta_y]
     * @return 3x3 rotation matrix Rgw (world to gravity frame)
     */
    Eigen::Matrix3d gravity_dir_to_rotation(const Eigen::Vector2d& gravity_dir) const;
};

/**
 * @brief Simple bias prior factor
 * 
 * Applies a zero-mean Gaussian prior on IMU biases:
 * cost = 0.5 * weight * ||bias - prior||^2
 */
class BiasPriorFactor : public ceres::SizedCostFunction<3, 3> {
public:
    BiasPriorFactor(const Eigen::Vector3d& prior, double weight)
        : prior_(prior), weight_(weight) {}

    virtual bool Evaluate(double const* const* parameters,
                         double* residuals,
                         double** jacobians) const override {
        const double* bias = parameters[0];
        
        // Residual: r = weight * (bias - prior)
        for (int i = 0; i < 3; ++i) {
            residuals[i] = weight_ * (bias[i] - prior_[i]);
        }
        
        // Jacobian w.r.t bias: dr/dbias = weight * I
        if (jacobians != nullptr && jacobians[0] != nullptr) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    jacobians[0][i * 3 + j] = (i == j) ? weight_ : 0.0;
                }
            }
        }
        
        return true;
    }

private:
    Eigen::Vector3d prior_;
    double weight_;
};

/**
 * @brief Velocity prior factor for IMU initialization
 */
class VelocityPriorFactor : public ceres::SizedCostFunction<3, 3> {
public:
    VelocityPriorFactor(const Eigen::Vector3d& prior, double weight)
        : prior_(prior), weight_(weight) {}

    virtual bool Evaluate(double const* const* parameters,
                         double* residuals,
                         double** jacobians) const override {
        
        // Extract velocity parameters
        Eigen::Map<const Eigen::Vector3d> velocity(parameters[0]);
        
        // Compute residual: sqrt(weight) * (velocity - prior)
        Eigen::Vector3d error = velocity - prior_;
        Eigen::Map<Eigen::Vector3d> residual(residuals);
        residual = sqrt(weight_) * error;
        
        // Compute Jacobian if requested
        if (jacobians != nullptr && jacobians[0] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian(jacobians[0]);
            jacobian = sqrt(weight_) * Eigen::Matrix3d::Identity();
        }
        
        return true;
    }

private:
    Eigen::Vector3d prior_;
    double weight_;
};

/**
 * @brief Accelerometer bias prior factor
 */
class AccelBiasPriorFactor : public ceres::SizedCostFunction<3, 3> {
public:
    AccelBiasPriorFactor(const Eigen::Vector3d& prior, double weight)
        : prior_(prior), weight_(weight) {}

    virtual bool Evaluate(double const* const* parameters,
                         double* residuals,
                         double** jacobians) const override {
        
        // Extract accel bias parameters
        Eigen::Map<const Eigen::Vector3d> accel_bias(parameters[0]);
        
        // Compute residual: sqrt(weight) * (bias - prior)
        Eigen::Vector3d error = accel_bias - prior_;
        Eigen::Map<Eigen::Vector3d> residual(residuals);
        residual = sqrt(weight_) * error;
        
        // Compute Jacobian if requested
        if (jacobians != nullptr && jacobians[0] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian(jacobians[0]);
            jacobian = sqrt(weight_) * Eigen::Matrix3d::Identity();
        }
        
        return true;
    }

private:
    Eigen::Vector3d prior_;
    double weight_;
};

/**
 * @brief Gyroscope bias prior factor
 */
class GyroBiasPriorFactor : public ceres::SizedCostFunction<3, 3> {
public:
    GyroBiasPriorFactor(const Eigen::Vector3d& prior, double weight)
        : prior_(prior), weight_(weight) {}

    virtual bool Evaluate(double const* const* parameters,
                         double* residuals,
                         double** jacobians) const override {
        
        // Extract gyro bias parameters
        Eigen::Map<const Eigen::Vector3d> gyro_bias(parameters[0]);
        
        // Compute residual: sqrt(weight) * (bias - prior)
        Eigen::Vector3d error = gyro_bias - prior_;
        Eigen::Map<Eigen::Vector3d> residual(residuals);
        residual = sqrt(weight_) * error;
        
        // Compute Jacobian if requested
        if (jacobians != nullptr && jacobians[0] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian(jacobians[0]);
            jacobian = sqrt(weight_) * Eigen::Matrix3d::Identity();
        }
        
        return true;
    }

private:
    Eigen::Vector3d prior_;
    double weight_;
};

/**
 * @brief Combined velocity and bias prior factor for IMU initialization
 */
class VelocityBiasPriorFactor : public ceres::SizedCostFunction<9, 9> {
public:
    VelocityBiasPriorFactor(const Eigen::VectorXd& prior, const Eigen::MatrixXd& information)
        : prior_(prior), information_(information) {}

    virtual bool Evaluate(double const* const* parameters,
                         double* residuals,
                         double** jacobians) const override {
        
        // Extract velocity+bias parameters [v(3), ba(3), bg(3)]
        Eigen::Map<const Eigen::VectorXd> velocity_bias(parameters[0], 9);
        
        // Compute residual: sqrt_info * (velocity_bias - prior)
        Eigen::VectorXd error = velocity_bias - prior_;
        
        // Apply square root information matrix
        Eigen::LLT<Eigen::MatrixXd> llt(information_);
        Eigen::MatrixXd sqrt_info = llt.matrixL().transpose();
        
        Eigen::Map<Eigen::VectorXd> residual(residuals, 9);
        residual = sqrt_info * error;
        
        // Compute Jacobian if requested
        if (jacobians != nullptr && jacobians[0] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor>> jacobian(jacobians[0]);
            jacobian = sqrt_info;
        }
        
        return true;
    }

private:
    Eigen::VectorXd prior_;
    Eigen::MatrixXd information_;
};

} // namespace factor
} // namespace lightweight_vio
