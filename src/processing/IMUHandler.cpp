/**
 * @file      IMUHandler.cpp
 * @brief     Implements IMU data preintegration and bias management
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-09-08
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "processing/IMUHandler.h"
#include "processing/Optimizer.h"
#include "database/MapPoint.h"
#include "util/EurocUtils.h"
#include "util/Config.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <ceres/ceres.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace lightweight_vio {

IMUPreintegration::IMUPreintegration() {
    reset();
}

void IMUPreintegration::reset() {
    delta_R = Eigen::Matrix3f::Identity();
    delta_V = Eigen::Vector3f::Zero();
    delta_P = Eigen::Vector3f::Zero();
    
    JRg = Eigen::Matrix3f::Zero();
    JVg = Eigen::Matrix3f::Zero();
    JVa = Eigen::Matrix3f::Zero();
    JPg = Eigen::Matrix3f::Zero();
    JPa = Eigen::Matrix3f::Zero();
    
    covariance = Eigen::Matrix<float, 15, 15>::Zero();
    
    gyro_bias = Eigen::Vector3f::Zero();
    accel_bias = Eigen::Vector3f::Zero();
    
    dt_total = 0.0;
}

bool IMUPreintegration::is_valid() const {
    return dt_total > 0.0 && !delta_R.hasNaN() && !delta_V.hasNaN() && !delta_P.hasNaN();
}

IMUHandler::IMUHandler() 
    : m_gyro_bias(Eigen::Vector3f::Zero())
    , m_accel_bias(Eigen::Vector3f::Zero())
    , m_gravity(0.0f, 0.0f, -9.81f)  // Default gravity pointing down
    , m_Rgw(Eigen::Matrix3f::Identity())
    , m_gravity_aligned(false)
    , m_initialized(false) {
    
    // Load noise parameters from config
    const auto& config = Config::getInstance();
    m_gyro_noise = static_cast<float>(config.m_gyro_noise_density);
    m_accel_noise = static_cast<float>(config.m_accel_noise_density);
    m_gyro_bias_noise = static_cast<float>(config.m_gyro_random_walk);
    m_accel_bias_noise = static_cast<float>(config.m_accel_random_walk);
    
    spdlog::info("[IMU_HANDLER] Initialized with config parameters:");
    spdlog::info("  - Gyro noise: {:.6e} rad/s/√Hz", m_gyro_noise);
    spdlog::info("  - Accel noise: {:.6e} m/s²/√Hz", m_accel_noise);
    spdlog::info("  - Gyro bias noise: {:.6e} rad/s²/√Hz", m_gyro_bias_noise);
    spdlog::info("  - Accel bias noise: {:.6e} m/s³/√Hz", m_accel_bias_noise);
}

void IMUHandler::reset() {
    m_gyro_bias = Eigen::Vector3f::Zero();
    m_accel_bias = Eigen::Vector3f::Zero();
    m_gravity = Eigen::Vector3f(0.0f, 0.0f, -9.81f);
    m_Rgw = Eigen::Matrix3f::Identity();
    m_gravity_aligned = false;
    m_initialized = false;
    
    spdlog::info("[IMU_HANDLER] Reset to initial state");
}

void IMUHandler::set_bias(const Eigen::Vector3f& gyro_bias, const Eigen::Vector3f& accel_bias) {
    m_gyro_bias = gyro_bias;
    m_accel_bias = accel_bias;
    
    spdlog::debug("[IMU_HANDLER] Updated bias - Gyro: ({:.6f}, {:.6f}, {:.6f}), Accel: ({:.6f}, {:.6f}, {:.6f})",
                 gyro_bias.x(), gyro_bias.y(), gyro_bias.z(),
                 accel_bias.x(), accel_bias.y(), accel_bias.z());
}

void IMUHandler::get_bias(Eigen::Vector3f& gyro_bias, Eigen::Vector3f& accel_bias) const {
    gyro_bias = m_gyro_bias;
    accel_bias = m_accel_bias;
}

std::shared_ptr<IMUPreintegration> IMUHandler::preintegrate(
    const std::vector<IMUData>& imu_measurements,
    double start_time,
    double end_time) {
    
    if (imu_measurements.empty()) {
        spdlog::warn("[IMU_HANDLER] No IMU measurements provided for preintegration");
        return nullptr;
    }
    
    auto preint = std::make_shared<IMUPreintegration>();
    preint->gyro_bias = m_gyro_bias;
    preint->accel_bias = m_accel_bias;
    
    // Filter measurements within time range
    std::vector<IMUData> filtered_measurements;
    for (const auto& imu : imu_measurements) {
        if (imu.timestamp >= start_time && imu.timestamp < end_time) {
            filtered_measurements.push_back(imu);
        }
    }
    
    if (filtered_measurements.empty()) {
        spdlog::warn("[IMU_HANDLER] No IMU measurements in time range [{:.6f}, {:.6f})", start_time, end_time);
        return nullptr;
    }
    
    // Integrate measurements WITHOUT gravity compensation (gravity is handled in optimization factors)
    bool use_gravity_compensation = false;  // Always false for proper preintegration
    
    for (size_t i = 0; i < filtered_measurements.size(); ++i) {
        float dt;
        if (i == 0) {
            dt = (filtered_measurements.size() > 1) ? 
                 static_cast<float>(filtered_measurements[1].timestamp - filtered_measurements[0].timestamp) : 0.005f;
        } else {
            dt = static_cast<float>(filtered_measurements[i].timestamp - filtered_measurements[i-1].timestamp);
        }
        dt = std::max(0.001f, std::min(dt, 0.02f));
        
        // Always use raw measurements for proper preintegration
        integrate_measurement(preint, filtered_measurements[i], dt);
        
        update_covariance(preint, dt);
        preint->dt_total += dt;
    }
    
    return preint;
}

void IMUHandler::integrate_measurement(
    std::shared_ptr<IMUPreintegration> preint,
    const IMUData& imu,
    float dt) {
    
    // Bias-corrected measurements
    Eigen::Vector3f gyro = imu.angular_vel - preint->gyro_bias;
    Eigen::Vector3f accel = imu.linear_accel - preint->accel_bias;
    
    // Current state
    Eigen::Matrix3f R = preint->delta_R;
    Eigen::Vector3f V = preint->delta_V;
    Eigen::Vector3f P = preint->delta_P;
    
    // Update rotation using Rodrigues formula
    Eigen::Vector3f omega_dt = gyro * dt;
    Eigen::Matrix3f dR = rodrigues(omega_dt);
    Eigen::Matrix3f Jr = right_jacobian(omega_dt);
    
    // Update Jacobians w.r.t gyro bias
    preint->JRg = -dR.transpose() * Jr * dt;
    preint->JVg = preint->JVg + preint->JVa * skew_symmetric(accel) * preint->JRg;
    preint->JPg = preint->JPg + preint->JPa * skew_symmetric(accel) * preint->JRg + preint->JVg * dt;
    
    // Update rotation
    preint->delta_R = R * dR;
    
    // Update velocity
    Eigen::Vector3f dV = R * accel * dt;
    preint->delta_V = V + dV;
    preint->JVa = preint->JVa + R * dt;  // Jacobian w.r.t accel bias
    
    // Update position  
    Eigen::Vector3f dP = V * dt + 0.5f * R * accel * dt * dt;
    preint->delta_P = P + dP;
    preint->JPa = preint->JPa + preint->JVa * dt + 0.5f * R * dt * dt;  // Jacobian w.r.t accel bias
}

void IMUHandler::integrate_measurement_with_gravity(
    std::shared_ptr<IMUPreintegration> preint,
    const IMUData& imu,
    float dt,
    const Eigen::Matrix3f& R_wb) {
    
    // Remove bias from measurements  
    Eigen::Vector3f gyro = imu.angular_vel - preint->gyro_bias;
    Eigen::Vector3f accel = imu.linear_accel - preint->accel_bias;
    
    // 🎯 중력 보상: body frame에서 중력 제거
    // 방정식: v_j_w - v_i_w = g_w*dt + R_wb*∫(a_compensated)dt
    // 따라서: a_compensated = a_measured - g_body, 여기서 g_body = R_wb^T * g_w
    // m_gravity = [0, 0, -9.81]이므로 실제로는 더해야 함
    Eigen::Vector3f gravity_body = R_wb.transpose() * m_gravity;
    Eigen::Vector3f accel_compensated = accel + gravity_body;

    // Current state
    Eigen::Matrix3f R = preint->delta_R;
    Eigen::Vector3f V = preint->delta_V;
    Eigen::Vector3f P = preint->delta_P;
    
    // Update rotation using Rodrigues formula
    Eigen::Vector3f omega_dt = gyro * dt;
    Eigen::Matrix3f dR = rodrigues(omega_dt);
    Eigen::Matrix3f Jr = right_jacobian(omega_dt);
    
    // Update Jacobians w.r.t gyro bias
    preint->JRg = -dR.transpose() * Jr * dt;
    preint->JVg = preint->JVg + preint->JVa * skew_symmetric(accel_compensated) * preint->JRg;
    preint->JPg = preint->JPg + preint->JPa * skew_symmetric(accel_compensated) * preint->JRg + preint->JVg * dt;
    
    // Update rotation
    preint->delta_R = R * dR;
    
    // Update velocity (중력 보상된 가속도 사용)
    Eigen::Vector3f dV = R * accel_compensated * dt;
    preint->delta_V = V + dV;
    preint->JVa = preint->JVa + R * dt;  // Jacobian w.r.t accel bias
    
    // Update position (중력 보상된 가속도 사용) 
    Eigen::Vector3f dP = V * dt + 0.5f * R * accel_compensated * dt * dt;
    preint->delta_P = P + dP;
    preint->JPa = preint->JPa + preint->JVa * dt + 0.5f * R * dt * dt;  // Jacobian w.r.t accel bias
}

void IMUHandler::update_covariance(std::shared_ptr<IMUPreintegration> preint, float dt) {
    // Noise covariance matrix
    Eigen::Matrix<float, 12, 12> Q = Eigen::Matrix<float, 12, 12>::Zero();
    
    // Gyroscope noise
    Q.block<3,3>(0,0) = Eigen::Matrix3f::Identity() * (m_gyro_noise * m_gyro_noise * dt);
    // Accelerometer noise  
    Q.block<3,3>(3,3) = Eigen::Matrix3f::Identity() * (m_accel_noise * m_accel_noise * dt);
    // Gyroscope bias random walk
    Q.block<3,3>(6,6) = Eigen::Matrix3f::Identity() * (m_gyro_bias_noise * m_gyro_bias_noise * dt);
    // Accelerometer bias random walk
    Q.block<3,3>(9,9) = Eigen::Matrix3f::Identity() * (m_accel_bias_noise * m_accel_bias_noise * dt);
    
    // State transition matrix F (simplified version)
    Eigen::Matrix<float, 15, 15> F = Eigen::Matrix<float, 15, 15>::Identity();
    
    // Noise mapping matrix G
    Eigen::Matrix<float, 15, 12> G = Eigen::Matrix<float, 15, 12>::Zero();
    G.block<3,3>(0,0) = preint->JRg;      // Rotation noise
    G.block<3,3>(3,3) = preint->JVa;      // Velocity noise  
    G.block<3,3>(6,6) = preint->JPa;      // Position noise
    G.block<3,3>(9,6) = Eigen::Matrix3f::Identity();   // Gyro bias noise
    G.block<3,3>(12,9) = Eigen::Matrix3f::Identity();  // Accel bias noise
    
    // Covariance propagation: P = F*P*F' + G*Q*G'
    preint->covariance = F * preint->covariance * F.transpose() + G * Q * G.transpose();
}

void IMUHandler::update_preintegration_with_bias(
    std::shared_ptr<IMUPreintegration> preint,
    const Eigen::Vector3f& delta_bg,
    const Eigen::Vector3f& delta_ba) {
    
    if (!preint || !preint->is_valid()) {
        spdlog::warn("[IMU_HANDLER] Invalid preintegration for bias update");
        return;
    }
    
    spdlog::debug("[IMU_HANDLER] Updating preintegration with bias change - dBg: ({:.6f}, {:.6f}, {:.6f}), dBa: ({:.6f}, {:.6f}, {:.6f})",
                 delta_bg.x(), delta_bg.y(), delta_bg.z(),
                 delta_ba.x(), delta_ba.y(), delta_ba.z());
    
    // Update using Jacobians (much faster than re-integration)
    preint->delta_R = preint->delta_R * rodrigues(preint->JRg * delta_bg);
    preint->delta_V = preint->delta_V + preint->JVg * delta_bg + preint->JVa * delta_ba;
    preint->delta_P = preint->delta_P + preint->JPg * delta_bg + preint->JPa * delta_ba;
    
    // Update bias
    preint->gyro_bias += delta_bg;
    preint->accel_bias += delta_ba;
}

void IMUHandler::estimate_initial_bias(
    const std::vector<IMUData>& imu_measurements,
    float gravity_magnitude) {
    
    if (imu_measurements.empty()) {
        spdlog::warn("[IMU_HANDLER] No IMU measurements for bias estimation");
        return;
    }
    
    spdlog::info("[IMU_HANDLER] Estimating initial bias from {} measurements", imu_measurements.size());
    
    // Estimate gyroscope bias (should be zero in static conditions)
    Eigen::Vector3f gyro_sum = Eigen::Vector3f::Zero();
    Eigen::Vector3f accel_sum = Eigen::Vector3f::Zero();
    
    for (const auto& imu : imu_measurements) {
        gyro_sum += imu.angular_vel;
        accel_sum += imu.linear_accel;
    }
    
    // Average as bias estimate
    m_gyro_bias = gyro_sum / static_cast<float>(imu_measurements.size());
    
    // 🎯 STEP 1: Estimate gravity direction from accelerometer measurements
    // In static conditions, accelerometer measures only gravity (+ bias)
    Eigen::Vector3f accel_mean = accel_sum / static_cast<float>(imu_measurements.size());
    
    // 🎯 STEP 2: Gravity direction = average accelerometer reading direction  
    Eigen::Vector3f gravity_direction = accel_mean.normalized();
    float measured_gravity_mag = accel_mean.norm();
    
    // 🎯 STEP 3: Update gravity vector (pointing opposite to measured acceleration)
    m_gravity = -gravity_direction * gravity_magnitude;
    
    // 🎯 STEP 4: Accelerometer bias = measured - expected gravity
    Eigen::Vector3f expected_accel = -m_gravity;  // Expected acceleration due to gravity
    m_accel_bias = accel_mean - expected_accel;
    
    m_initialized = true;
    
    spdlog::info("[IMU_HANDLER] ✅ Initial bias estimated:");
    spdlog::info("  - Gravity direction: ({:.6f}, {:.6f}, {:.6f})", 
                 gravity_direction.x(), gravity_direction.y(), gravity_direction.z());
    spdlog::info("  - Measured gravity magnitude: {:.6f} m/s²", measured_gravity_mag);
    spdlog::info("  - Gyro bias: ({:.6f}, {:.6f}, {:.6f})", 
                 m_gyro_bias.x(), m_gyro_bias.y(), m_gyro_bias.z());
    spdlog::info("  - Accel bias: ({:.6f}, {:.6f}, {:.6f})", 
                 m_accel_bias.x(), m_accel_bias.y(), m_accel_bias.z());
}

bool IMUHandler::estimate_gravity_with_stereo_constraints(
    const std::vector<Frame*>& frames,
    const std::vector<IMUData>& all_imu_data,
    float gravity_magnitude) {
    
    if (frames.size() < 3) {
        spdlog::warn("[IMU_HANDLER] Need at least 3 frames for gravity estimation");
        return false;
    }
    
    spdlog::info("[IMU_HANDLER] 🎯 Starting gravity estimation with {} frames", frames.size());
    
    // Step 1: Create preintegrations (without gravity compensation first)
    std::vector<std::shared_ptr<IMUPreintegration>> preintegrations;
    for (size_t i = 0; i < frames.size() - 1; i++) {
        double start_time = static_cast<double>(frames[i]->get_timestamp()) / 1e9;
        double end_time = static_cast<double>(frames[i + 1]->get_timestamp()) / 1e9;
        
        auto preint = preintegrate(all_imu_data, start_time, end_time);
        if (!preint) continue;
        preintegrations.push_back(preint);
    }
    
    // No need to check preintegration count - if we have enough frames, 
    // we should have enough preintegrations
    
    const int N = frames.size();
    const int M = preintegrations.size();
    
    // Step 2: Setup linear system for gravity estimation
    // Variables: [g_x, g_y, g_z, v0_x, v0_y, v0_z, ..., v_{N-1}_x, v_{N-1}_y, v_{N-1}_z]
    const int num_unknowns = 3 + 3 * N;
    const int num_equations = 3 * M + 3 * N;
    
    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(num_equations, num_unknowns);
    Eigen::VectorXf b = Eigen::VectorXf::Zero(num_equations);
    int eq_idx = 0;
    
    // IMU constraints: v_{i+1} - v_i - g*dt = R_wb_i * delta_V / dt
    for (int i = 0; i < M; i++) {
        auto preint = preintegrations[i];
        Eigen::Matrix3f R_wb = frames[i]->get_Twb().block<3,3>(0,0);
        float dt = preint->dt_total;
        Eigen::Vector3f imu_vel = R_wb * preint->delta_V / dt;
        
        for (int axis = 0; axis < 3; axis++) {
            // Gravity coefficient: -dt
            A(eq_idx + axis, axis) = -dt;
            // v_i coefficient: -1
            A(eq_idx + axis, 3 + 3*i + axis) = -1.0f;
            // v_{i+1} coefficient: +1  
            A(eq_idx + axis, 3 + 3*(i+1) + axis) = 1.0f;
            // RHS: IMU velocity
            b(eq_idx + axis) = imu_vel(axis);
        }
        eq_idx += 3;
    }
    
    // Stereo visual constraints: smooth velocity assumption
    const float stereo_weight = 0.1f;  // Weaker constraint
    for (int i = 0; i < N; i++) {
        Eigen::Vector3f stereo_vel = Eigen::Vector3f::Zero();
        
        // Estimate velocity from neighboring frames
        if (i > 0 && i < N-1) {
            // Central difference
            Eigen::Matrix4f T_prev = frames[i-1]->get_Twb();
            Eigen::Matrix4f T_next = frames[i+1]->get_Twb();
            Eigen::Vector3f pos_diff = T_next.block<3,1>(0,3) - T_prev.block<3,1>(0,3);
            float dt_total = preintegrations[i-1]->dt_total + preintegrations[i]->dt_total;
            stereo_vel = pos_diff / dt_total;
        }
        
        for (int axis = 0; axis < 3; axis++) {
            A(eq_idx + axis, 3 + 3*i + axis) = stereo_weight;
            b(eq_idx + axis) = stereo_weight * stereo_vel(axis);
        }
        eq_idx += 3;
    }
    
    // Step 3: Solve linear system
    Eigen::MatrixXf AtA = A.transpose() * A;
    Eigen::VectorXf Atb = A.transpose() * b;
    AtA += 1e-6f * Eigen::MatrixXf::Identity(num_unknowns, num_unknowns);
    
    Eigen::VectorXf solution = AtA.lu().solve(Atb);
    Eigen::Vector3f estimated_gravity = solution.segment<3>(0);
    
    // Step 4: Normalize and set gravity
    if (estimated_gravity.norm() < 1.0f) {
        spdlog::error("[IMU_HANDLER] Invalid gravity estimation");
        return false;
    }
    
    m_gravity = estimated_gravity.normalized() * gravity_magnitude;
    m_initialized = true;
    
    // 🎯 Calculate Rgw (World-to-Gravity transformation matrix)
    // =====================================================================
    // Purpose: Transform World coordinate system to Gravity-aligned coordinate system
    // Notation: Rab = transformation from frame b to frame a
    // Rgw = World → Gravity transformation (transforms World frame vectors to Gravity frame)
    // 
    // Why this transformation is needed:
    // 1. IMU optimization stability: In gravity frame, gravity is always [0,0,-9.81] (known constant)
    // 2. Coordinate standardization: All datasets share same reference frame regardless of initial orientation
    // 
    // Algorithm: Calculate rotation matrix that aligns current gravity direction with ideal [0,0,-1]
    // =====================================================================
    
    Eigen::Vector3f gravity_direction = m_gravity.normalized();  // Current estimated gravity direction (normalized)
    Eigen::Vector3f gravity_ideal(0.0f, 0.0f, -1.0f);          // Ideal gravity direction (downward Z-axis)
    
    // Calculate rotation axis and angle for Rodrigues formula
    Eigen::Vector3f rotation_axis = gravity_ideal.cross(gravity_direction);  // Rotation axis = ideal × current
    float rotation_axis_norm = rotation_axis.norm();                         // Rotation axis magnitude (proportional to sin(θ))
    float cos_angle = gravity_ideal.dot(gravity_direction);                  // cos(θ) = dot product of unit vectors
    
    Eigen::Matrix3f Rgw;
    if (rotation_axis_norm < 1e-6f) {
        // Case 1: Gravity vectors are nearly parallel (aligned or anti-aligned)
        if (cos_angle > 0.0f) {
            // Gravity already aligned with ideal direction
            Rgw = Eigen::Matrix3f::Identity();
        } else {
            // 180 degree rotation needed - choose any perpendicular axis
            Eigen::Vector3f perp_axis(1.0f, 0.0f, 0.0f);  // Default to X-axis
            if (std::abs(gravity_direction.x()) > 0.9f) {
                perp_axis = Eigen::Vector3f(0.0f, 1.0f, 0.0f);  // Use Y-axis if X is nearly parallel
            }
            Rgw = rodrigues(perp_axis * M_PI);  // 180 degree rotation around perpendicular axis
        }
    } else {
        // Case 2: General rotation using Rodrigues formula
        // Calculate rotation angle and apply axis-angle rotation
        float angle = std::acos(std::clamp(cos_angle, -1.0f, 1.0f));  // Clamp to handle numerical errors
        Eigen::Vector3f normalized_axis = rotation_axis / rotation_axis_norm;
        Eigen::Vector3f rotation_vector = normalized_axis * angle;
        Rgw = rodrigues(rotation_vector);  // Convert axis-angle to rotation matrix
    }
    
    // Store the transformation matrix for future use
    m_Rgw = Rgw;
    m_gravity_aligned = true;  // Mark that we have computed the transformation
    
    spdlog::info("[IMU_HANDLER] ✅ Gravity estimation completed:");
    spdlog::info("  Estimated gravity: ({:.4f}, {:.4f}, {:.4f}) m/s²", 
                 estimated_gravity.x(), estimated_gravity.y(), estimated_gravity.z());
    spdlog::info("  Final gravity: ({:.4f}, {:.4f}, {:.4f}) m/s²", 
                 m_gravity.x(), m_gravity.y(), m_gravity.z());
    spdlog::info("  Gravity direction: ({:.4f}, {:.4f}, {:.4f})", 
                 gravity_direction.x(), gravity_direction.y(), gravity_direction.z());
    spdlog::info("  Gravity magnitude: {:.4f} m/s²", m_gravity.norm());
    spdlog::info("  Rgw (World-to-Gravity rotation matrix):");
    spdlog::info("    [{:.6f}, {:.6f}, {:.6f}]", Rgw(0,0), Rgw(0,1), Rgw(0,2));
    spdlog::info("    [{:.6f}, {:.6f}, {:.6f}]", Rgw(1,0), Rgw(1,1), Rgw(1,2));
    spdlog::info("    [{:.6f}, {:.6f}, {:.6f}]", Rgw(2,0), Rgw(2,1), Rgw(2,2));
    
    // Verify the transformation
    Eigen::Vector3f transformed_gravity = Rgw * gravity_direction;
    spdlog::info("  Verification - Transformed gravity: ({:.4f}, {:.4f}, {:.4f})", 
                 transformed_gravity.x(), transformed_gravity.y(), transformed_gravity.z());
    
    return true;
}

bool IMUHandler::debug_velocity_comparison(
    const std::vector<Frame*>& frames,
    const std::vector<IMUData>& all_imu_data) {
    
    if (frames.size() < 2) {
        spdlog::warn("[IMU_HANDLER] Need at least 2 frames for velocity comparison");
        return false;
    }
    
    if (!m_initialized || m_gravity.norm() < 9.0f) {
        spdlog::warn("[IMU_HANDLER] Gravity not estimated, run gravity estimation first");
        return false;
    }
    
    spdlog::info("=== 🎯 Velocity Comparison: Stereo VO vs Gravity-Compensated IMU ===");
    spdlog::info("Gravity: ({:.4f}, {:.4f}, {:.4f}) m/s²", m_gravity.x(), m_gravity.y(), m_gravity.z());
    spdlog::info("Current IMU Biases - Gyro: ({:.6f}, {:.6f}, {:.6f}), Accel: ({:.6f}, {:.6f}, {:.6f})", 
                 m_gyro_bias.x(), m_gyro_bias.y(), m_gyro_bias.z(),
                 m_accel_bias.x(), m_accel_bias.y(), m_accel_bias.z());
    
    float total_error = 0.0f;
    int count = 0;
    
    for (size_t i = 0; i < std::min(frames.size() - 1, size_t(5)); ++i) {
        Frame* frame1 = frames[i];
        Frame* frame2 = frames[i + 1];
        
        double start_time = static_cast<double>(frame1->get_timestamp()) / 1e9;
        double end_time = static_cast<double>(frame2->get_timestamp()) / 1e9;
        float dt = static_cast<float>(end_time - start_time);
        
        if (dt <= 0.001f) continue;
        
        // Stereo velocity
        
        // IMU velocity (raw preintegration without gravity compensation)
        auto preint = preintegrate(all_imu_data, start_time, end_time);
        if (!preint) continue;
        
        count++;
    }
    
    if (count > 0) {
        float avg_error = total_error / count;
        spdlog::info("Average velocity error: {:.4f} m/s", avg_error);
        return avg_error < 0.5f;  // Reasonable threshold
    }
    
    return false;
}



// ==================== HELPER FUNCTIONS ====================

Eigen::Matrix3f IMUHandler::rodrigues(const Eigen::Vector3f& w) const {
    float theta = w.norm();
    if (theta < 1e-6f) {
        return Eigen::Matrix3f::Identity() + skew_symmetric(w);
    }
    
    Eigen::Vector3f axis = w / theta;
    float c = std::cos(theta);
    float s = std::sin(theta);
    
    return c * Eigen::Matrix3f::Identity() + 
           s * skew_symmetric(axis) + 
           (1.0f - c) * axis * axis.transpose();
}

Eigen::Matrix3f IMUHandler::right_jacobian(const Eigen::Vector3f& w) const {
    float theta = w.norm();
    if (theta < 1e-6f) {
        return Eigen::Matrix3f::Identity() - 0.5f * skew_symmetric(w);
    }
    
    Eigen::Vector3f axis = w / theta;
    float c = std::cos(theta);
    float s = std::sin(theta);
    
    return s / theta * Eigen::Matrix3f::Identity() + 
           (1.0f - c) / theta * skew_symmetric(axis) + 
           (theta - s) / theta * axis * axis.transpose();
}

Eigen::Matrix3f IMUHandler::skew_symmetric(const Eigen::Vector3f& v) const {
    Eigen::Matrix3f skew = Eigen::Matrix3f::Zero();
    skew(0, 1) = -v(2);
    skew(0, 2) =  v(1);
    skew(1, 0) =  v(2);
    skew(1, 2) = -v(0);
    skew(2, 0) = -v(1);
    skew(2, 1) =  v(0);
    return skew;
}

bool IMUHandler::transform_to_gravity_frame(
    const std::vector<Frame*>& keyframes,
    std::vector<std::shared_ptr<MapPoint>>& map_points,
    Eigen::Matrix4f& T_gw) {

    if (!m_gravity_aligned) {
        spdlog::error("[IMU_HANDLER] Gravity alignment not computed, call estimate_gravity_with_stereo_constraints first");
        return false;
    }
    
    if (keyframes.empty()) {
        spdlog::warn("[IMU_HANDLER] No keyframes to transform");
        return false;
    }
    
    spdlog::info("[IMU_HANDLER] 🌍 Transforming {} keyframes and {} map points to gravity-aligned frame", 
                 keyframes.size(), map_points.size());
    
    // Transform all keyframe poses


    int ii =0;
    for (Frame* frame : keyframes) {
        if (!frame) continue;
        
        // Get current pose T_wb (world-to-body)
        Eigen::Matrix4f T_wb = frame->get_Twb();
        
        // Transform pose using SE(3) gravity transformation T_gb = T_gw * T_wb
        T_gw = Eigen::Matrix4f::Identity();
        T_gw.block<3,3>(0,0) = m_Rgw;
        
        Eigen::Matrix4f T_gb = T_gw * T_wb;

        std::cout<<"Check T_gw here!! \n"<<T_gw<<std::endl;

        if(ii == 0)
        {
            std::cout<<"Check T_wb original here!! \n"<<T_wb<<std::endl;
            std::cout<<"Check T_gb transformed here!! \n"<<T_gb<<std::endl;
            ii++;
        }
        
        // Set transformed pose
        frame->set_Twb(T_gb);
        
        // Transform velocity to gravity frame
        Eigen::Vector3f velocity_world = frame->get_velocity();
        Eigen::Vector3f velocity_gravity = m_Rgw * velocity_world;
        frame->set_velocity(velocity_gravity);
        
        // Frame transformation completed (reduced logging)
    }
    
    // Transform all map points
    int transformed_count = 0;
    for (auto& map_point : map_points) {
        if (!map_point) continue;
        
        // Get current 3D position in world frame
        Eigen::Vector3f pos_world = map_point->get_position();
        
        // Transform to gravity frame: pos_gravity = Rgw * pos_world
        Eigen::Vector3f pos_gravity = m_Rgw * pos_world;
        
        // Set transformed position
        map_point->set_position(pos_gravity);
        transformed_count++;
    }
    
    spdlog::info("[IMU_HANDLER] ✅ Successfully transformed {} keyframes and {} map points to gravity frame", 
                 keyframes.size(), transformed_count);
    spdlog::info("[IMU_HANDLER] Gravity vector in new frame: ({:.4f}, {:.4f}, {:.4f}) m/s²", 
                 0.0f, 0.0f, -m_gravity.norm());
    
    return true;
}

bool IMUHandler::apply_optimization_results(
    const std::vector<Frame*>& optimized_keyframes,
    const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& optimized_biases,
    const std::vector<Eigen::Vector3f>& optimized_velocities) {
    
    if (optimized_keyframes.size() != optimized_biases.size() || 
        optimized_keyframes.size() != optimized_velocities.size()) {
        spdlog::error("[IMU_HANDLER] Size mismatch: keyframes={}, biases={}, velocities={}", 
                     optimized_keyframes.size(), optimized_biases.size(), optimized_velocities.size());
        return false;
    }
    
    spdlog::info("[IMU_HANDLER] 🎯 Applying optimization results to {} keyframes", optimized_keyframes.size());
    
    // Apply results to each keyframe
    for (size_t i = 0; i < optimized_keyframes.size(); ++i) {
        Frame* frame = optimized_keyframes[i];
        if (!frame) continue;
        
        const auto& bias_pair = optimized_biases[i];
        const Eigen::Vector3f& gyro_bias = bias_pair.first;
        const Eigen::Vector3f& accel_bias = bias_pair.second;
        const Eigen::Vector3f& velocity = optimized_velocities[i];
        
        // Set optimized values
        frame->set_gyro_bias(gyro_bias);
        frame->set_accel_bias(accel_bias);
        frame->set_velocity(velocity);
        
        // Frame optimization applied (reduced logging)
    }
    
    // Update IMUHandler's global bias with latest optimized values (summary only)
    if (!optimized_keyframes.empty() && !optimized_biases.empty()) {
        const auto& latest_bias = optimized_biases.back();
        m_gyro_bias = latest_bias.first;
        m_accel_bias = latest_bias.second;
        
        spdlog::info("[IMU_HANDLER] ✅ Updated global bias from optimization");
    }
    
    return true;
}

bool IMUHandler::initialize_first_keyframe_from_second(
    Frame* first_keyframe,
    const Frame* second_keyframe) {
    
    if (!first_keyframe || !second_keyframe) {
        spdlog::error("[IMU_HANDLER] Null keyframe provided for initialization");
        return false;
    }
    
    // Copy bias and velocity from second keyframe to first
    first_keyframe->set_gyro_bias(second_keyframe->get_gyro_bias());
    first_keyframe->set_accel_bias(second_keyframe->get_accel_bias());
    first_keyframe->set_velocity(second_keyframe->get_velocity());
    
    spdlog::info("[IMU_HANDLER] ✅ Initialized first keyframe {} from second keyframe {}", 
                 first_keyframe->get_frame_id(), second_keyframe->get_frame_id());
    spdlog::info("  - Gyro bias: ({:.6f}, {:.6f}, {:.6f})", 
                 first_keyframe->get_gyro_bias().x(), first_keyframe->get_gyro_bias().y(), first_keyframe->get_gyro_bias().z());
    spdlog::info("  - Accel bias: ({:.6f}, {:.6f}, {:.6f})", 
                 first_keyframe->get_accel_bias().x(), first_keyframe->get_accel_bias().y(), first_keyframe->get_accel_bias().z());
    spdlog::info("  - Velocity: ({:.6f}, {:.6f}, {:.6f})", 
                 first_keyframe->get_velocity().x(), first_keyframe->get_velocity().y(), first_keyframe->get_velocity().z());
    
    return true;
}

bool IMUHandler::inherit_bias_from_keyframe(
    Frame* new_frame,
    const Frame* reference_keyframe) {
    
    if (!new_frame || !reference_keyframe) {
        spdlog::error("[IMU_HANDLER] Null frame provided for bias inheritance");
        return false;
    }
    
    // Copy bias from reference keyframe
    new_frame->set_gyro_bias(reference_keyframe->get_gyro_bias());
    new_frame->set_accel_bias(reference_keyframe->get_accel_bias());
    
    // Update IMUHandler's global bias as well
    m_gyro_bias = reference_keyframe->get_gyro_bias();
    m_accel_bias = reference_keyframe->get_accel_bias();
    
    spdlog::debug("[IMU_HANDLER] Frame {} inherited bias from keyframe {} - Gyro: ({:.6f},{:.6f},{:.6f}), Accel: ({:.6f},{:.6f},{:.6f})",
                 new_frame->get_frame_id(), reference_keyframe->get_frame_id(),
                 m_gyro_bias.x(), m_gyro_bias.y(), m_gyro_bias.z(),
                 m_accel_bias.x(), m_accel_bias.y(), m_accel_bias.z());
    
    return true;
}

void IMUHandler::compute_rgw_transformation() {
    if (m_gravity.norm() < 1.0f) {
        spdlog::warn("[IMU_HANDLER] Invalid gravity vector for Rgw computation");
        return;
    }
    
    // 🎯 Calculate Rgw (World-to-Gravity transformation matrix)
    // =====================================================================
    // Purpose: Transform World coordinate system to Gravity-aligned coordinate system
    // Notation: Rab = transformation from frame b to frame a
    // Rgw = World → Gravity transformation (transforms World frame vectors to Gravity frame)
    // 
    // Algorithm: Calculate rotation matrix that aligns current gravity direction with ideal [0,0,-1]
    // =====================================================================
    
    Eigen::Vector3f gravity_direction = m_gravity.normalized();  // Current estimated gravity direction (normalized)
    Eigen::Vector3f gravity_ideal(0.0f, 0.0f, -1.0f);          // Ideal gravity direction (downward Z-axis)
    
    // Calculate rotation axis and angle for Rodrigues formula
    Eigen::Vector3f rotation_axis = gravity_ideal.cross(gravity_direction);  // Rotation axis = ideal × current
    float rotation_axis_norm = rotation_axis.norm();                         // Rotation axis magnitude (proportional to sin(θ))
    float cos_angle = gravity_ideal.dot(gravity_direction);                  // cos(θ) = dot product of unit vectors
    
    if (rotation_axis_norm < 1e-6f) {
        // Case 1: Gravity vectors are nearly parallel (aligned or anti-aligned)
        if (cos_angle > 0.0f) {
            // Gravity already aligned with ideal direction
            m_Rgw = Eigen::Matrix3f::Identity();
        } else {
            // 180 degree rotation needed - choose any perpendicular axis
            Eigen::Vector3f perp_axis(1.0f, 0.0f, 0.0f);  // Default to X-axis
            if (std::abs(gravity_direction.x()) > 0.9f) {
                perp_axis = Eigen::Vector3f(0.0f, 1.0f, 0.0f);  // Use Y-axis if X is nearly parallel
            }
            m_Rgw = rodrigues(perp_axis * M_PI);  // 180 degree rotation around perpendicular axis
        }
    } else {
        // Case 2: General rotation using Rodrigues formula
        // Calculate rotation angle and apply axis-angle rotation
        float angle = std::acos(std::clamp(cos_angle, -1.0f, 1.0f));  // Clamp to handle numerical errors
        Eigen::Vector3f normalized_axis = rotation_axis / rotation_axis_norm;
        Eigen::Vector3f rotation_vector = normalized_axis * angle;
        m_Rgw = rodrigues(rotation_vector);  // Convert axis-angle to rotation matrix
    }
    
    m_gravity_aligned = true;  // Mark that we have computed the transformation
    
    spdlog::info("[IMU_HANDLER] ✅ Computed Rgw transformation matrix:");
    spdlog::info("  Current gravity: ({:.4f}, {:.4f}, {:.4f}) m/s²", 
                 m_gravity.x(), m_gravity.y(), m_gravity.z());
    spdlog::info("  Gravity direction: ({:.4f}, {:.4f}, {:.4f})", 
                 gravity_direction.x(), gravity_direction.y(), gravity_direction.z());
    spdlog::info("  Rgw matrix:");
    spdlog::info("    [{:.6f}, {:.6f}, {:.6f}]", m_Rgw(0,0), m_Rgw(0,1), m_Rgw(0,2));
    spdlog::info("    [{:.6f}, {:.6f}, {:.6f}]", m_Rgw(1,0), m_Rgw(1,1), m_Rgw(1,2));
    spdlog::info("    [{:.6f}, {:.6f}, {:.6f}]", m_Rgw(2,0), m_Rgw(2,1), m_Rgw(2,2));
    
    // Verify the transformation
    Eigen::Vector3f transformed_gravity = m_Rgw * gravity_direction;
    spdlog::info("  Verification - Transformed gravity: ({:.4f}, {:.4f}, {:.4f})", 
                 transformed_gravity.x(), transformed_gravity.y(), transformed_gravity.z());
}



Eigen::Vector3f IMUHandler::get_gravity_compensated_velocity(
    const Frame* frame,
    float dt) const {
    
    if (!frame || dt <= 0.0f) {
        spdlog::warn("[IMU_HANDLER] Invalid parameters for gravity compensation");
        return Eigen::Vector3f::Zero();
    }
    
    // Get current velocity (may include gravity effects)
    Eigen::Vector3f velocity = frame->get_velocity();
    
    // In gravity-aligned frame, gravity is [0, 0, -9.81]
    // Remove gravity effects: v_real = v_measured - g * dt
    // But since our velocity is already preintegrated and optimized, 
    // we don't need to subtract gravity*dt here
    
    // The velocity from optimization should already be the real motion velocity
    // because the InertialGravityFactor handles gravity properly
    
    return velocity;  // Already gravity-compensated from optimization
}

bool IMUHandler::update_preintegrations_with_optimized_bias(
    const std::vector<Frame*>& frames,
    const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& optimized_biases) {
    
    if (frames.size() != optimized_biases.size()) {
        spdlog::error("[IMU_HANDLER] Size mismatch: frames={}, biases={}", 
                     frames.size(), optimized_biases.size());
        return false;
    }
    
    spdlog::info("[IMU_HANDLER] 🔄 Updating preintegrations with optimized bias for {} frames", frames.size());
    
    int updated_count = 0;
    for (size_t i = 0; i < frames.size(); ++i) {
        Frame* frame = frames[i];
        if (!frame) continue;
        
        // Try to get preintegration data from different sources
        auto preint = frame->get_imu_preintegration_from_last_keyframe();
        if (!preint) {
            preint = frame->get_imu_preintegration_from_last_frame();
        }
        
        if (!preint) {
            spdlog::debug("[IMU_HANDLER] Frame {}: No preintegration data available", frame->get_frame_id());
            continue;
        }
        
        const auto& bias_pair = optimized_biases[i];
        const Eigen::Vector3f& new_gyro_bias = bias_pair.first;
        const Eigen::Vector3f& new_accel_bias = bias_pair.second;
        
        // Calculate bias change
        Eigen::Vector3f delta_bg = new_gyro_bias - preint->gyro_bias;
        Eigen::Vector3f delta_ba = new_accel_bias - preint->accel_bias;
        
        // Update preintegration with new bias using Jacobians (fast method)
        update_preintegration_with_bias(preint, delta_bg, delta_ba);
        
        updated_count++;
        
        // Preintegration updated (reduced logging)
        updated_count++;
    }
    
    spdlog::info("[IMU_HANDLER] ✅ Updated {} preintegrations with optimized bias", updated_count);
    return updated_count > 0;
}

void IMUHandler::set_gravity_aligned_coordinate_system() {
    if (!m_gravity_aligned) {
        spdlog::warn("[IMU_HANDLER] Gravity alignment not yet computed, cannot set aligned coordinate system");
        return;
    }
    
    // After transforming to gravity-aligned frame, set Rgw to Identity
    // This indicates we are now working in the gravity-aligned coordinate system
    m_Rgw = Eigen::Matrix3f::Identity();
    
    // Update gravity vector to point in standard downward direction
    m_gravity = Eigen::Vector3f(0.0f, 0.0f, -9.81f);
    
    spdlog::info("[IMU_HANDLER] ✅ Set coordinate system to gravity-aligned frame");
    spdlog::info("  - Rgw is now Identity (no further transformation needed)");
    spdlog::info("  - Gravity vector: (0.0, 0.0, -9.81) m/s²");
}

} // namespace lightweight_vio
