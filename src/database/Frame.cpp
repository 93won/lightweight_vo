#include "Frame.h"
#include "MapPoint.h"
#include "../util/Config.h"
#include <algorithm>
#include <iostream>

namespace lightweight_vio {

Frame::Frame(long long timestamp, int frame_id)
    : m_timestamp(timestamp)
    , m_frame_id(frame_id)
    , m_rotation(Eigen::Matrix3f::Identity())
    , m_translation(Eigen::Vector3f::Zero())
    , m_is_keyframe(false)
    , m_fx(500.0), m_fy(500.0)  // Default focal lengths
    , m_cx(320.0), m_cy(240.0)  // Default principal point
{
    // Initialize default distortion coefficients (no distortion)
    m_distortion_coeffs = {0.0, 0.0, 0.0, 0.0, 0.0};
}

Frame::Frame(long long timestamp, int frame_id, double fx, double fy, double cx, double cy, 
             const std::vector<double>& distortion_coeffs)
    : m_timestamp(timestamp)
    , m_frame_id(frame_id)
    , m_rotation(Eigen::Matrix3f::Identity())
    , m_translation(Eigen::Vector3f::Zero())
    , m_is_keyframe(false)
    , m_fx(fx), m_fy(fy)
    , m_cx(cx), m_cy(cy)
    , m_distortion_coeffs(distortion_coeffs)
{
}

void Frame::set_pose(const Eigen::Matrix3f& rotation, const Eigen::Vector3f& translation) {
    m_rotation = rotation;
    m_translation = translation;
}

void Frame::add_feature(std::shared_ptr<Feature> feature) {
    m_features.push_back(feature);
    m_feature_id_to_index[feature->get_feature_id()] = m_features.size() - 1;
    // Add corresponding null map point and outlier flag
    m_map_points.push_back(nullptr);
    m_outlier_flags.push_back(false);
}

void Frame::remove_feature(int feature_id) {
    auto it = m_feature_id_to_index.find(feature_id);
    if (it != m_feature_id_to_index.end()) {
        size_t index = it->second;
        m_features.erase(m_features.begin() + index);
        m_map_points.erase(m_map_points.begin() + index);
        m_outlier_flags.erase(m_outlier_flags.begin() + index);
        m_feature_id_to_index.erase(it);
        update_feature_index();
    }
}

std::shared_ptr<Feature> Frame::get_feature(int feature_id) {
    auto it = m_feature_id_to_index.find(feature_id);
    if (it != m_feature_id_to_index.end()) {
        return m_features[it->second];
    }
    return nullptr;
}

std::shared_ptr<const Feature> Frame::get_feature(int feature_id) const {
    auto it = m_feature_id_to_index.find(feature_id);
    if (it != m_feature_id_to_index.end()) {
        return m_features[it->second];
    }
    return nullptr;
}

void Frame::extract_features(int max_features) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (m_left_image.empty()) {
        std::cerr << "Cannot extract features: left image is empty" << std::endl;
        return;
    }

    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(m_left_image, corners, max_features, m_quality_level, m_min_distance);

    static int global_feature_id = 0;
    for (const auto& corner : corners) {
        auto feature = std::make_shared<Feature>(global_feature_id++, corner);
        add_feature(feature);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    const Config& config = Config::getInstance();
    if (config.isTimingEnabled()) {
        std::cout << "[TIMING] Feature extraction: " << duration.count() / 1000.0 << " ms | "
                  << "Extracted " << corners.size() << " features" << std::endl;
    }
}

cv::Mat Frame::draw_features() const {
    cv::Mat display_image;
    if (m_left_image.channels() == 1) {
        cv::cvtColor(m_left_image, display_image, cv::COLOR_GRAY2BGR);
    } else {
        display_image = m_left_image.clone();
    }

    for (const auto& feature : m_features) {
        if (feature->is_valid()) {
            const cv::Point2f& pt = feature->get_pixel_coord();
            int track_count = feature->get_track_count();
            float len = std::min(1.0f, 1.0f * track_count / 20.0f);
            cv::Scalar color(255 * (1 - len), 0, 255 * len);
            cv::circle(display_image, pt, 2, color, 2);
        }
    }

    return display_image;
}

cv::Mat Frame::draw_stereo_matches() const {
    if (!is_stereo()) {
        std::cout << "Cannot draw stereo matches: not a stereo frame" << std::endl;
        return cv::Mat();
    }

    // Create side-by-side display
    cv::Mat left_display, right_display;
    if (m_left_image.channels() == 1) {
        cv::cvtColor(m_left_image, left_display, cv::COLOR_GRAY2BGR);
        cv::cvtColor(m_right_image, right_display, cv::COLOR_GRAY2BGR);
    } else {
        left_display = m_left_image.clone();
        right_display = m_right_image.clone();
    }

    // Create combined image
    cv::Mat combined_image;
    cv::hconcat(left_display, right_display, combined_image);
    
    int right_offset = m_left_image.cols;

    // Draw features and matches
    for (const auto& feature : m_features) {
        if (feature->is_valid()) {
            const cv::Point2f& left_pt = feature->get_pixel_coord();
            
            // Draw left feature (green circle)
            cv::circle(combined_image, left_pt, 3, cv::Scalar(0, 255, 0), 2);
            
            if (feature->has_stereo_match()) {
                const cv::Point2f& right_pt = feature->get_right_coord();
                
                // Check if stereo match is valid (not (-1, -1))
                if (right_pt.x >= 0 && right_pt.y >= 0) {
                    cv::Point2f right_pt_shifted(right_pt.x + right_offset, right_pt.y);
                    
                    // Draw right feature (red circle)
                    cv::circle(combined_image, right_pt_shifted, 3, cv::Scalar(0, 0, 255), 2);
                    
                    // Draw matching line (yellow)
                    cv::line(combined_image, left_pt, right_pt_shifted, cv::Scalar(0, 255, 255), 1);
                    
                    // Show 3D point info instead of disparity
                    if (feature->has_3d_point()) {
                        Eigen::Vector3f pt3d = feature->get_3d_point();
                        std::string depth_str = cv::format("%.2fm", pt3d[2]);
                        cv::putText(combined_image, depth_str, 
                                   cv::Point(left_pt.x + 5, left_pt.y - 5), 
                                   cv::FONT_HERSHEY_SIMPLEX, 0.4, 
                                   cv::Scalar(255, 255, 255), 1);
                    }
                }
            }
        }
    }

    // Add labels
    cv::putText(combined_image, "Left Camera", cv::Point(10, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    cv::putText(combined_image, "Right Camera", cv::Point(right_offset + 10, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);

    return combined_image;
}

cv::Mat Frame::draw_rectified_stereo_matches() const {
    if (!is_stereo()) {
        std::cout << "Cannot draw normalized stereo matches: not a stereo frame" << std::endl;
        return cv::Mat();
    }

    // Create side-by-side display showing normalized coordinate space visualization
    cv::Mat left_display, right_display;
    if (m_left_image.channels() == 1) {
        cv::cvtColor(m_left_image, left_display, cv::COLOR_GRAY2BGR);
        cv::cvtColor(m_right_image, right_display, cv::COLOR_GRAY2BGR);
    } else {
        left_display = m_left_image.clone();
        right_display = m_right_image.clone();
    }

    // Create combined image
    cv::Mat combined_image;
    cv::hconcat(left_display, right_display, combined_image);
    
    int right_offset = m_left_image.cols;
    int valid_stereo_matches = 0;

    // Draw normalized features and matches
    for (const auto& feature : m_features) {
        if (feature->is_valid()) {
            // Use original pixel coordinates for visualization
            cv::Point2f left_px = feature->get_pixel_coord();
            
            // Draw left feature (green circle)
            cv::circle(combined_image, left_px, 3, cv::Scalar(0, 255, 0), 2);
            
            if (feature->has_stereo_match()) {
                cv::Point2f right_px = feature->get_right_coord();
                
                // Check if stereo match is valid
                if (right_px.x >= 0 && right_px.y >= 0) {
                    cv::Point2f right_px_shifted(right_px.x + right_offset, right_px.y);
                    
                    // Draw right feature (red circle)
                    cv::circle(combined_image, right_px_shifted, 3, cv::Scalar(0, 0, 255), 2);
                    
                    // Draw matching line (yellow)
                    cv::line(combined_image, left_px, right_px_shifted, cv::Scalar(0, 255, 255), 1);
                    
                    // Show 3D point depth if available
                    if (feature->has_3d_point()) {
                        Eigen::Vector3f pt3d = feature->get_3d_point();
                        std::string depth_str = cv::format("%.2fm", pt3d[2]);
                        cv::putText(combined_image, depth_str, 
                                   cv::Point(left_px.x + 5, left_px.y - 5), 
                                   cv::FONT_HERSHEY_SIMPLEX, 0.4, 
                                   cv::Scalar(255, 255, 255), 1);
                    }
                    
                    // Show normalized coordinates for debugging
                    Eigen::Vector2f norm = feature->get_normalized_coord();
                    if (norm[0] != 0.0f || norm[1] != 0.0f) { // Check if normalized coord is valid
                        std::string norm_str = cv::format("(%.2f,%.2f)", norm[0], norm[1]);
                        cv::putText(combined_image, norm_str, 
                                   cv::Point(left_px.x + 5, left_px.y + 15), 
                                   cv::FONT_HERSHEY_SIMPLEX, 0.3, 
                                   cv::Scalar(0, 255, 255), 1);
                    }
                    
                    valid_stereo_matches++;
                }
            }
        }
    }

    // Add labels
    cv::putText(combined_image, "Left (Normalized)", cv::Point(10, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    cv::putText(combined_image, "Right (Normalized)", cv::Point(right_offset + 10, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    
    // Show statistics
    cv::putText(combined_image, cv::format("Valid matches: %d", valid_stereo_matches), 
               cv::Point(10, combined_image.rows - 10), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

    return combined_image;
}

cv::Mat Frame::draw_tracks(const Frame& previous_frame) const {
    cv::Mat display_image = draw_features();

    for (const auto& feature : m_features) {
        if (!feature->is_valid()) continue;

        auto prev_feature = previous_frame.get_feature(feature->get_feature_id());
        if (prev_feature && prev_feature->is_valid()) {
            cv::line(display_image, 
                    prev_feature->get_pixel_coord(), 
                    feature->get_pixel_coord(), 
                    cv::Scalar(0, 255, 0), 1);
        }
    }

    return display_image;
}

void Frame::update_feature_index() {
    m_feature_id_to_index.clear();
    for (size_t i = 0; i < m_features.size(); ++i) {
        m_feature_id_to_index[m_features[i]->get_feature_id()] = i;
    }
}

bool Frame::is_in_border(const cv::Point2f& point, int border_size) const {
    int img_x = cvRound(point.x);
    int img_y = cvRound(point.y);
    return border_size <= img_x && img_x < m_left_image.cols - border_size && 
           border_size <= img_y && img_y < m_left_image.rows - border_size;
}

void Frame::compute_stereo_matches() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (!is_stereo()) {
        std::cout << "Cannot compute stereo matches: right image not available" << std::endl;
        return;
    }

    std::vector<cv::Point2f> left_pts, right_pts;
    std::vector<uchar> status;
    std::vector<float> err;

    // Extract feature points from left image
    for (const auto& feature : m_features) {
        if (feature->is_valid()) {
            left_pts.push_back(feature->get_pixel_coord());
        }
    }

    if (left_pts.empty()) {
        std::cout << "No features to match in stereo" << std::endl;
        return;
    }

    // Perform optical flow tracking from left to right image with improved parameters
    int window_size = Config::getInstance().getWindowSize();
    cv::calcOpticalFlowPyrLK(m_left_image, m_right_image, left_pts, right_pts, 
                            status, err, cv::Size(window_size, window_size), 
                            Config::getInstance().getMaxLevel(),
                            Config::getInstance().getTermCriteria(),
                            0, Config::getInstance().getMinEigenThreshold()); // Lower eigenvalue threshold for better tracking

    int matches_found = 0;
    
    // For unrectified stereo, we need more sophisticated matching
    // First, try to estimate fundamental matrix from initial matches
    std::vector<cv::Point2f> good_left_pts, good_right_pts;
    
    // Collect initial matches with very loose criteria
    size_t feature_idx = 0;
    for (auto& feature : m_features) {
        if (feature->is_valid() && feature_idx < status.size()) {
            if (status[feature_idx] && err[feature_idx] < Config::getInstance().getStereoErrorThreshold()) { // Very loose error threshold
                good_left_pts.push_back(left_pts[feature_idx]);
                good_right_pts.push_back(right_pts[feature_idx]);
            }
            feature_idx++;
        }
    }
    
    cv::Mat fundamental_matrix;
    std::vector<uchar> inlier_mask;
    
    if (good_left_pts.size() >= 8) {
        // Estimate fundamental matrix with RANSAC
        fundamental_matrix = cv::findFundamentalMat(
            good_left_pts, good_right_pts, cv::FM_RANSAC, 
            Config::getInstance().getFundamentalThreshold(), 
            Config::getInstance().getFundamentalConfidence(), inlier_mask
        );
        
        const Config& config = Config::getInstance();
        if (config.isDebugOutputEnabled()) {
            std::cout << "Fundamental matrix estimated from " << cv::countNonZero(inlier_mask) 
                      << "/" << good_left_pts.size() << " initial matches" << std::endl;
        }
    }
    
    // Now apply matches with epipolar constraint
    feature_idx = 0;
    for (auto& feature : m_features) {
        if (feature->is_valid() && feature_idx < status.size()) {
            if (status[feature_idx] && err[feature_idx] < Config::getInstance().getStereoErrorThreshold()) {
                cv::Point2f left_pt = left_pts[feature_idx];
                cv::Point2f right_pt = right_pts[feature_idx];
                
                bool is_valid_match = true;
                
                // Check epipolar constraint if fundamental matrix is available
                if (!fundamental_matrix.empty()) {
                    // Convert points to homogeneous coordinates with correct type
                    cv::Mat left_homo = (cv::Mat_<double>(3, 1) << left_pt.x, left_pt.y, 1.0);
                    cv::Mat right_homo = (cv::Mat_<double>(3, 1) << right_pt.x, right_pt.y, 1.0);
                    
                    // Ensure fundamental matrix is double type
                    cv::Mat F_double;
                    if (fundamental_matrix.type() != CV_64F) {
                        fundamental_matrix.convertTo(F_double, CV_64F);
                    } else {
                        F_double = fundamental_matrix;
                    }
                    
                    // Compute epipolar error: x2^T * F * x1
                    cv::Mat epipolar_error = right_homo.t() * F_double * left_homo;
                    double error = std::abs(epipolar_error.at<double>(0, 0));
                    
                    // Reject if epipolar error is too large
                    if (error > Config::getInstance().getEpipolarThreshold()) {
                        is_valid_match = false;
                    }
                }
                
                // Additional basic checks (remove disparity-based checks)
                float y_diff = std::abs(left_pt.y - right_pt.y);
                
                // Reject if y-coordinate difference is too large (basic sanity check)
                if (y_diff > Config::getInstance().getMaxYDifference()) {
                    is_valid_match = false;
                }
                
                if (is_valid_match) {
                    feature->set_stereo_match(right_pt, -1.0f); // No disparity stored
                    matches_found++;
                } else {
                    // Invalid match - reset stereo match data
                    feature->set_stereo_match(cv::Point2f(-1, -1), -1.0f);
                }
            } else {
                // No match found - set invalid disparity
                feature->set_stereo_match(cv::Point2f(-1, -1), -1.0f);
            }
            feature_idx++;
        } else if (feature->is_valid()) {
            // Feature is valid but no corresponding tracking result - set invalid disparity
            feature->set_stereo_match(cv::Point2f(-1, -1), -1.0f);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    if (Config::getInstance().isTimingEnabled()) {
        std::cout << "[TIMING] Stereo matching: " << duration.count() / 1000.0 << " ms | "
                  << "Matched " << matches_found << "/" << left_pts.size() << " features"
                  << " (using epipolar constraint, no disparity)" << std::endl;
    }
}

void Frame::undistort_features() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize outlier flags for all features
    initialize_outlier_flags();
    
    const Config& config = Config::getInstance();
    cv::Mat left_K = config.getLeftCameraMatrix();
    cv::Mat left_D = config.getLeftDistCoeffs();
    cv::Mat right_K = config.getRightCameraMatrix();
    cv::Mat right_D = config.getRightDistCoeffs();
    
    if (left_K.empty() || left_D.empty()) {
        std::cerr << "Camera calibration not available for undistortion" << std::endl;
        return;
    }
    
    // Convert to Eigen for easier computation  
    Eigen::Matrix3d K_left_eigen;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            K_left_eigen(i, j) = left_K.at<double>(i, j);
        }
    }
    
    // Process all features - convert to normalized coordinates only
    for (auto& feature : m_features) {
        if (feature->is_valid()) {
            // Get original pixel coordinate
            cv::Point2f pixel_pt = feature->get_pixel_coord();
            
            // Method 1: Direct undistortion to normalized coordinates (no rectification)
            std::vector<cv::Point2f> distorted_pts = {pixel_pt};
            std::vector<cv::Point2f> undistorted_pts;
            
            // Undistort to normalized coordinates (output is already normalized)
            cv::undistortPoints(distorted_pts, undistorted_pts, left_K, left_D);
            
            // Store undistorted coordinate (for compatibility, though we mainly use normalized)
            feature->set_undistorted_coord(undistorted_pts[0]);
            
            // Store normalized coordinate (same as undistorted in this case)
            Eigen::Vector2f normalized(undistorted_pts[0].x, undistorted_pts[0].y);
            feature->set_normalized_coord(normalized);
            
            // For stereo matches, we don't need rectification anymore
            // The triangulation will handle the geometric relationship directly
            if (feature->has_stereo_match()) {
                cv::Point2f right_pixel = feature->get_right_coord();
                
                // Check if stereo match is valid
                if (right_pixel.x >= 0 && right_pixel.y >= 0) {
                    // For right camera, undistort to normalized coordinates
                    std::vector<cv::Point2f> right_distorted = {right_pixel};
                    std::vector<cv::Point2f> right_undistorted;
                    
                    cv::undistortPoints(right_distorted, right_undistorted, right_K, right_D);
                    
                    // Also calculate normalized coordinates for right camera
                    Eigen::Vector2f right_normalized(right_undistorted[0].x, right_undistorted[0].y);
                    
                    // Store right undistorted coordinate and normalized coordinate
                    feature->set_undistorted_stereo_match(right_undistorted[0], right_normalized, -1.0f);
                } else {
                    // Invalid stereo match
                    feature->set_undistorted_stereo_match(cv::Point2f(-1, -1), Eigen::Vector2f(-1,-1), -1.0f);
                }
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Count valid stereo matches (no disparity constraint)
    int valid_stereo_matches = 0;
    for (const auto& feature : m_features) {
        if (feature->is_valid() && feature->has_stereo_match()) {
            cv::Point2f right_undist = feature->get_right_undistorted_coord();
            if (right_undist.x >= 0 && right_undist.y >= 0) {
                valid_stereo_matches++;
            }
        }
    }
    
    if (config.isTimingEnabled()) {
        std::cout << "[TIMING] Feature undistortion to normalized coords: " << duration.count() / 1000.0 << " ms | "
                  << "Processed " << m_features.size() << " features | "
                  << "Valid stereo matches: " << valid_stereo_matches << std::endl;
    }
}

void Frame::triangulate_stereo_points() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    const Config& config = Config::getInstance();
    cv::Mat left_K = config.getLeftCameraMatrix();
    cv::Mat left_D = config.getLeftDistCoeffs();
    cv::Mat right_K = config.getRightCameraMatrix();
    cv::Mat right_D = config.getRightDistCoeffs();
    cv::Mat T_rl = config.getLeftToRightTransform();  // T_rl: left to right transform (following T_ab = b->a convention)
    
    if (left_K.empty() || right_K.empty() || T_rl.empty()) {
        std::cerr << "Camera calibration not available for triangulation" << std::endl;
        return;
    }
    
    // Extract rotation and translation (T_rl: left to right transform)
    // This directly gives us left-to-right transformation for triangulation
    cv::Mat R_lr = T_rl(cv::Rect(0, 0, 3, 3));  // Left to right rotation  
    cv::Mat t_lr = T_rl(cv::Rect(3, 0, 1, 3));  // Left to right translation
    
    // Convert to Eigen for easier computation
    Eigen::Matrix3d R_eigen, K_left_eigen, K_right_eigen;
    Eigen::Vector3d t_eigen;
    
    // Convert CV matrices to Eigen (manual conversion)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            R_eigen(i, j) = R_lr.at<double>(i, j);  // Use left-to-right rotation
            K_left_eigen(i, j) = left_K.at<double>(i, j);
            K_right_eigen(i, j) = right_K.at<double>(i, j);
        }
    }
    // Handle translation vector separately (3x1)
    t_eigen << t_lr.at<double>(0,0), t_lr.at<double>(1,0), t_lr.at<double>(2,0);  // Use left-to-right translation
    
    // Debug: Print camera baseline
    double baseline = t_eigen.norm();
    if (config.isDebugOutputEnabled()) {
        std::cout << "Camera baseline: " << baseline << " meters" << std::endl;
        std::cout << "Translation vector: (" << t_eigen[0] << ", " << t_eigen[1] << ", " << t_eigen[2] << ")" << std::endl;
    }
    
    // Counters for debugging
    int triangulated_count = 0;
    int depth_rejected = 0;
    int reprojection_rejected = 0;
    int total_stereo_matches = 0;
    int normalized_fail_count = 0;
    std::vector<double> depth_values;
    
    for (auto& feature : m_features) {
        if (feature->is_valid() && feature->has_stereo_match()) {
            total_stereo_matches++;
            
            // Get pre-calculated normalized coordinates (more accurate)
            Eigen::Vector2f left_norm_2d = feature->get_normalized_coord();
            Eigen::Vector2f right_norm_2d = feature->get_right_normalized_coord();
            
            // Check if stereo match is valid
            if (right_norm_2d[0] == -1.0f || right_norm_2d[1] == -1.0f) {
                continue;
            }
            
            // Convert to 3D homogeneous coordinates for DLT
            Eigen::Vector3d left_normalized(left_norm_2d[0], left_norm_2d[1], 1.0);
            Eigen::Vector3d right_normalized(right_norm_2d[0], right_norm_2d[1], 1.0);
            
            // Triangulation using SVD
            // Setup design matrix: A * X = 0
            Eigen::Matrix4d A;
            
            // Left camera projection matrix is [I | 0] (identity pose)
            Eigen::Matrix<double, 3, 4> P_left;
            P_left.setZero();
            P_left.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
            
            // Right camera projection matrix using T_rl (left-to-right transform)
            // T_rl directly gives us left-to-right transformation
            Eigen::Matrix<double, 3, 4> P_right;
            P_right.block<3,3>(0,0) = R_eigen;  // R_lr (left-to-right rotation)
            P_right.block<3,1>(0,3) = t_eigen;  // t_lr (left-to-right translation)
            
            // Build constraint equations: normalized_point^T * [P * X] = 0
            A.row(0) = left_normalized[0] * P_left.row(2) - P_left.row(0);
            A.row(1) = left_normalized[1] * P_left.row(2) - P_left.row(1);
            A.row(2) = right_normalized[0] * P_right.row(2) - P_right.row(0);
            A.row(3) = right_normalized[1] * P_right.row(2) - P_right.row(1);
            
            // Solve using SVD
            Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
            Eigen::Vector4d X_h = svd.matrixV().col(3);
            
            // Check homogeneous coordinate
            if (std::abs(X_h[3]) < 1e-6) {
                normalized_fail_count++;
                continue;
            }
            
            // Convert to 3D point in left camera frame
            X_h /= X_h[3];
            Eigen::Vector3d pos_3d = X_h.head<3>();
            
            depth_values.push_back(pos_3d[2]);
            
            // Debug: Print depth values to see what we're getting
            if (config.isDebugOutputEnabled() && depth_values.size() <= 10) {
                // std::cout << "Triangulated depth: " << pos_3d[2] 
                //           << ", 3D point: (" << pos_3d[0] << ", " << pos_3d[1] << ", " << pos_3d[2] << ")" 
                //           << ", left_norm: (" << left_norm_2d[0] << ", " << left_norm_2d[1] << ")"
                //           << ", right_norm: (" << right_norm_2d[0] << ", " << right_norm_2d[1] << ")" << std::endl;
            }
            
            // Check depth range (positive depth in front of camera)
            if (pos_3d[2] < config.getMinDepth() || pos_3d[2] > config.getMaxDepth()) {
                depth_rejected++;
                // if (config.isDebugOutputEnabled() && depth_rejected <= 5) {
                //     std::cout << "Depth rejected: " << pos_3d[2] << " (range: " 
                //               << config.getMinDepth() << " - " << config.getMaxDepth() << ")" << std::endl;
                // }
                continue;
            }
            
            // Reprojection error check using normalized coordinates
            // Project 3D point back to left camera (should match left_normalized)
            Eigen::Vector3d reproj_left = pos_3d; // Already in left camera frame
            reproj_left /= reproj_left[2]; // Normalize by depth
            
            // Project 3D point to right camera using T_rl (left-to-right)
            Eigen::Vector3d pos_right = R_eigen * pos_3d + t_eigen;
            if (pos_right[2] <= 0) { // Check positive depth in right camera
                depth_rejected++;
                continue;
            }
            Eigen::Vector3d reproj_right = pos_right / pos_right[2];
            
            // Calculate reprojection errors in normalized coordinates
            double left_error = (left_normalized.head<2>() - reproj_left.head<2>()).norm();
            double right_error = (right_normalized.head<2>() - reproj_right.head<2>()).norm();
            double max_error = std::max(left_error, right_error);
            
            // Use normalized coordinate reprojection threshold
            double max_reproj_error = config.getMaxReprojectionError() / std::min(K_left_eigen(0,0), K_left_eigen(1,1));
            
            if (max_error > max_reproj_error) {
                reprojection_rejected++;
                continue;
            }
            
            // Success! Store the 3D point in left camera frame
            Eigen::Vector3f point3D_float = pos_3d.cast<float>();
            feature->set_3d_point(point3D_float);
            
            // Update normalized coordinates in feature (using left camera)
            Eigen::Vector2f normalized_2d(left_normalized[0], left_normalized[1]);
            feature->set_normalized_coord(normalized_2d);
            
            triangulated_count++;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    if (config.isTimingEnabled()) {
        std::cout << "[TIMING] Normalized stereo triangulation: " << duration.count() / 1000.0 << " ms | "
                  << "Triangulated " << triangulated_count << " 3D points from " << total_stereo_matches 
                  << " stereo matches (depth rejected: " << depth_rejected 
                  << ", reprojection rejected: " << reprojection_rejected 
                  << ", normalization failed: " << normalized_fail_count << ")" << std::endl;
    }
}

void Frame::initialize_map_points() {
    m_map_points.clear();
    m_map_points.resize(m_features.size(), nullptr);
}

void Frame::set_map_point(int feature_index, std::shared_ptr<MapPoint> map_point) {
    if (feature_index >= 0 && feature_index < static_cast<int>(m_map_points.size())) {
        m_map_points[feature_index] = map_point;
    }
}

std::shared_ptr<MapPoint> Frame::get_map_point(int feature_index) const {
    if (feature_index >= 0 && feature_index < static_cast<int>(m_map_points.size())) {
        return m_map_points[feature_index];
    }
    return nullptr;
}

bool Frame::has_map_point(int feature_index) const {
    if (feature_index >= 0 && feature_index < static_cast<int>(m_map_points.size())) {
        return m_map_points[feature_index] != nullptr;
    }
    return false;
}

// Outlier flag management
void Frame::set_outlier_flag(int feature_index, bool is_outlier) {
    if (feature_index >= 0 && feature_index < static_cast<int>(m_outlier_flags.size())) {
        m_outlier_flags[feature_index] = is_outlier;
    }
}

bool Frame::get_outlier_flag(int feature_index) const {
    if (feature_index >= 0 && feature_index < static_cast<int>(m_outlier_flags.size())) {
        return m_outlier_flags[feature_index];
    }
    return false; // Default to not outlier if index is invalid
}

void Frame::initialize_outlier_flags() {
    m_outlier_flags.assign(m_features.size(), false);
}

// Camera parameter management
void Frame::set_camera_intrinsics(double fx, double fy, double cx, double cy) {
    m_fx = fx;
    m_fy = fy;
    m_cx = cx;
    m_cy = cy;
}

void Frame::get_camera_intrinsics(double& fx, double& fy, double& cx, double& cy) const {
    fx = m_fx;
    fy = m_fy;
    cx = m_cx;
    cy = m_cy;
}

void Frame::set_distortion_coeffs(const std::vector<double>& distortion_coeffs) {
    m_distortion_coeffs = distortion_coeffs;
}

cv::Point2f Frame::undistort_point(const cv::Point2f& distorted_point) const {
    if (m_distortion_coeffs.empty() || 
        (m_distortion_coeffs.size() >= 5 && 
         std::abs(m_distortion_coeffs[0]) < 1e-6 && 
         std::abs(m_distortion_coeffs[1]) < 1e-6)) {
        return distorted_point; // No significant distortion correction needed
    }

    // Convert to normalized coordinates
    double x = (distorted_point.x - m_cx) / m_fx;
    double y = (distorted_point.y - m_cy) / m_fy;

    // Iterative undistortion (Newton-Raphson method)
    if (m_distortion_coeffs.size() >= 5) {
        double k1 = m_distortion_coeffs[0];
        double k2 = m_distortion_coeffs[1];
        double p1 = m_distortion_coeffs[2];
        double p2 = m_distortion_coeffs[3];
        double k3 = m_distortion_coeffs[4];

        // Initial guess
        double x_u = x;
        double y_u = y;

        // Iterative correction (typically 5 iterations are enough)
        for (int iter = 0; iter < 5; ++iter) {
            double r2 = x_u*x_u + y_u*y_u;
            double r4 = r2*r2;
            double r6 = r4*r2;

            // Radial distortion
            double radial_factor = 1.0 + k1*r2 + k2*r4 + k3*r6;
            
            // Tangential distortion
            double dx = 2.0*p1*x_u*y_u + p2*(r2 + 2.0*x_u*x_u);
            double dy = p1*(r2 + 2.0*y_u*y_u) + 2.0*p2*x_u*y_u;

            // Distorted coordinates
            double x_d = x_u * radial_factor + dx;
            double y_d = y_u * radial_factor + dy;

            // Correction
            x_u = x_u - (x_d - x);
            y_u = y_u - (y_d - y);
        }

        x = x_u;
        y = y_u;
    }

    // Convert back to pixel coordinates
    return cv::Point2f(x * m_fx + m_cx, y * m_fy + m_cy);
}

} // namespace lightweight_vio
