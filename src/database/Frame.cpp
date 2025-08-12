#include "Frame.h"
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
{
}

void Frame::set_pose(const Eigen::Matrix3f& rotation, const Eigen::Vector3f& translation) {
    m_rotation = rotation;
    m_translation = translation;
}

void Frame::add_feature(std::shared_ptr<Feature> feature) {
    m_features.push_back(feature);
    m_feature_id_to_index[feature->get_feature_id()] = m_features.size() - 1;
}

void Frame::remove_feature(int feature_id) {
    auto it = m_feature_id_to_index.find(feature_id);
    if (it != m_feature_id_to_index.end()) {
        size_t index = it->second;
        m_features.erase(m_features.begin() + index);
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
                    
                    // Show disparity value (use undistorted disparity if available)
                    float disparity = feature->get_undistorted_disparity();
                    if (disparity <= 0) {
                        disparity = feature->get_stereo_disparity(); // fallback to original disparity
                    }
                    
                    std::string disp_str = cv::format("%.1f", disparity);
                    cv::putText(combined_image, disp_str, 
                               cv::Point(left_pt.x + 5, left_pt.y - 5), 
                               cv::FONT_HERSHEY_SIMPLEX, 0.4, 
                               cv::Scalar(255, 255, 255), 1);
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
        std::cout << "Cannot draw rectified stereo matches: not a stereo frame" << std::endl;
        return cv::Mat();
    }

    // Create side-by-side display of rectified images
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
    int valid_rectified_matches = 0;

    // Draw rectified features and matches
    for (const auto& feature : m_features) {
        if (feature->is_valid()) {
            // Use rectified (undistorted) coordinates for drawing
            cv::Point2f left_rect = feature->get_undistorted_coord();
            
            // Draw left rectified feature (green circle)
            cv::circle(combined_image, left_rect, 3, cv::Scalar(0, 255, 0), 2);
            
            if (feature->has_stereo_match()) {
                cv::Point2f right_rect = feature->get_right_undistorted_coord();
                
                // Check if rectified stereo match is valid
                if (right_rect.x >= 0 && right_rect.y >= 0 && feature->get_undistorted_disparity() > 0) {
                    cv::Point2f right_rect_shifted(right_rect.x + right_offset, right_rect.y);
                    
                    // Draw right rectified feature (red circle)
                    cv::circle(combined_image, right_rect_shifted, 3, cv::Scalar(0, 0, 255), 2);
                    
                    // Draw matching line (yellow) - should be horizontal for rectified stereo
                    cv::line(combined_image, left_rect, right_rect_shifted, cv::Scalar(0, 255, 255), 1);
                    
                    // Show rectified disparity value
                    float disparity = feature->get_undistorted_disparity();
                    std::string disp_str = cv::format("%.1f", disparity);
                    cv::putText(combined_image, disp_str, 
                               cv::Point(left_rect.x + 5, left_rect.y - 5), 
                               cv::FONT_HERSHEY_SIMPLEX, 0.4, 
                               cv::Scalar(255, 255, 255), 1);
                    
                    // Show Y difference (should be very small)
                    float y_diff = std::abs(left_rect.y - right_rect.y);
                    float max_y_threshold = Config::getInstance().getMaxRectifiedYDifference();
                    if (y_diff > max_y_threshold) { // Highlight if Y difference is large
                        cv::putText(combined_image, cv::format("Y:%.1f", y_diff), 
                                   cv::Point(left_rect.x + 5, left_rect.y + 15), 
                                   cv::FONT_HERSHEY_SIMPLEX, 0.3, 
                                   cv::Scalar(0, 0, 255), 1); // Red text for large Y diff
                    }
                    
                    valid_rectified_matches++;
                }
            }
        }
    }

    // Add labels
    cv::putText(combined_image, "Left Rectified", cv::Point(10, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    cv::putText(combined_image, "Right Rectified", cv::Point(right_offset + 10, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    
    // Show statistics
    cv::putText(combined_image, cv::format("Valid matches: %d", valid_rectified_matches), 
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
                
                // Additional basic checks
                float disparity = left_pt.x - right_pt.x;
                float y_diff = std::abs(left_pt.y - right_pt.y);
                
                if (disparity < Config::getInstance().getMinDisparity() || 
                    disparity > Config::getInstance().getMaxDisparity()) {
                    is_valid_match = false;
                }
                
                // Reject if y-coordinate difference is too large (even for unrectified stereo)
                if (y_diff > Config::getInstance().getMaxYDifference()) {
                    is_valid_match = false;
                }
                
                if (is_valid_match) {
                    feature->set_stereo_match(right_pt, disparity);
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
                  << " (using epipolar constraint for unrectified stereo)" << std::endl;
    }
}

void Frame::undistort_features() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    const Config& config = Config::getInstance();
    cv::Mat left_K = config.getLeftCameraMatrix();
    cv::Mat left_D = config.getLeftDistCoeffs();
    cv::Mat right_K = config.getRightCameraMatrix();
    cv::Mat right_D = config.getRightDistCoeffs();
    cv::Mat T_lr = config.getLeftToRightTransform();
    
    if (left_K.empty() || left_D.empty() || right_K.empty() || right_D.empty() || T_lr.empty()) {
        std::cerr << "Camera calibration not available for stereo rectification" << std::endl;
        return;
    }
    
    // Extract rotation and translation for stereo rectification
    cv::Mat R = T_lr(cv::Rect(0, 0, 3, 3));
    cv::Mat t = T_lr(cv::Rect(3, 0, 1, 3));
    
    // Image size
    cv::Size image_size(m_left_image.cols, m_left_image.rows);
    
    // Compute stereo rectification matrices
    cv::Mat R1, R2, P1, P2, Q;
    cv::Mat map1_left, map2_left, map1_right, map2_right;
    
    cv::stereoRectify(left_K, left_D, right_K, right_D, image_size, R, t,
                      R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1, image_size);
    
    // Create rectification maps
    cv::initUndistortRectifyMap(left_K, left_D, R1, P1, image_size, CV_32FC1, map1_left, map2_left);
    cv::initUndistortRectifyMap(right_K, right_D, R2, P2, image_size, CV_32FC1, map1_right, map2_right);
    
    // Process all features
    for (auto& feature : m_features) {
        if (feature->is_valid()) {
            // Rectify left feature point
            std::vector<cv::Point2f> left_distorted = {feature->get_pixel_coord()};
            std::vector<cv::Point2f> left_rectified;
            cv::undistortPoints(left_distorted, left_rectified, left_K, left_D, R1, P1);
            
            feature->set_undistorted_coord(left_rectified[0]);
            
            // Compute normalized coordinates using rectified camera matrix P1
            cv::Point2f rect_pt = left_rectified[0];
            double fx = P1.at<double>(0, 0);
            double fy = P1.at<double>(1, 1);
            double cx = P1.at<double>(0, 2);
            double cy = P1.at<double>(1, 2);
            
            double normalized_x = (static_cast<double>(rect_pt.x) - cx) / fx;
            double normalized_y = (static_cast<double>(rect_pt.y) - cy) / fy;
            Eigen::Vector2f normalized(static_cast<float>(normalized_x), static_cast<float>(normalized_y));
            feature->set_normalized_coord(normalized);
            
            // If stereo match exists, rectify right point and compute disparity
            if (feature->has_stereo_match()) {
                std::vector<cv::Point2f> right_distorted = {feature->get_right_coord()};
                std::vector<cv::Point2f> right_rectified;
                cv::undistortPoints(right_distorted, right_rectified, right_K, right_D, R2, P2);
                
                // Check if rectification was successful
                cv::Point2f left_rect = left_rectified[0];
                cv::Point2f right_rect = right_rectified[0];
                
                // Check Y difference in rectified coordinates (should be very small now)
                float rectified_y_diff = std::abs(left_rect.y - right_rect.y);
                float max_rectified_y_diff = static_cast<float>(config.getMaxRectifiedYDifference());
                
                if (rectified_y_diff > max_rectified_y_diff) {
                    // Y difference too large even after rectification - invalidate stereo match
                    feature->set_stereo_match(cv::Point2f(-1, -1), -1.0f);
                    feature->set_undistorted_stereo_match(cv::Point2f(-1, -1), -1.0f);
                } else {
                    // Calculate rectified disparity (should be positive for proper stereo)
                    float rectified_disparity = left_rect.x - right_rect.x;
                    
                    if (rectified_disparity > 0.0f && 
                        rectified_disparity >= static_cast<float>(config.getMinDisparity()) &&
                        rectified_disparity <= static_cast<float>(config.getMaxDisparity())) {
                        // Store rectified right coordinate and disparity
                        feature->set_undistorted_stereo_match(right_rect, rectified_disparity);
                    } else {
                        // Invalid disparity - invalidate stereo match
                        feature->set_stereo_match(cv::Point2f(-1, -1), -1.0f);
                        feature->set_undistorted_stereo_match(cv::Point2f(-1, -1), -1.0f);
                    }
                }
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Count valid rectified stereo matches
    int valid_rectified_matches = 0;
    for (const auto& feature : m_features) {
        if (feature->is_valid() && feature->has_stereo_match() && 
            feature->get_undistorted_disparity() > 0) {
            valid_rectified_matches++;
        }
    }
    
    if (config.isTimingEnabled()) {
        std::cout << "[TIMING] Stereo rectification: " << duration.count() / 1000.0 << " ms | "
                  << "Rectified " << m_features.size() << " features | "
                  << "Valid rectified stereo matches: " << valid_rectified_matches << std::endl;
    }
}

void Frame::triangulate_stereo_points() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    const Config& config = Config::getInstance();
    cv::Mat left_K = config.getLeftCameraMatrix();
    cv::Mat left_D = config.getLeftDistCoeffs();
    cv::Mat right_K = config.getRightCameraMatrix();
    cv::Mat right_D = config.getRightDistCoeffs();
    cv::Mat T_lr = config.getLeftToRightTransform();
    
    if (left_K.empty() || right_K.empty() || T_lr.empty()) {
        std::cerr << "Camera calibration not available for triangulation" << std::endl;
        return;
    }
    
    // Extract rotation and translation for stereo rectification
    cv::Mat R = T_lr(cv::Rect(0, 0, 3, 3));
    cv::Mat t = T_lr(cv::Rect(3, 0, 1, 3));
    
    // Image size
    cv::Size image_size(m_left_image.cols, m_left_image.rows);
    
    // Compute stereo rectification matrices (same as in undistort_features)
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(left_K, left_D, right_K, right_D, image_size, R, t,
                      R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1, image_size);
    
    // Convert rectified projection matrices to Eigen
    Eigen::Matrix<double, 3, 4> P1_eigen, P2_eigen;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            P1_eigen(i, j) = P1.at<double>(i, j);
            P2_eigen(i, j) = P2.at<double>(i, j);
        }
    }
    
    // Counters for debugging
    int triangulated_count = 0;
    int depth_rejected = 0;
    int reprojection_rejected = 0;
    int total_stereo_matches = 0;
    int homogeneous_fail_count = 0;
    std::vector<double> depth_values;
    
    for (auto& feature : m_features) {
        if (feature->is_valid() && feature->has_stereo_match()) {
            total_stereo_matches++;
            
            // Get rectified pixel coordinates (these are the "undistorted" coordinates from rectification)
            cv::Point2f left_pt = feature->get_undistorted_coord();
            cv::Point2f right_pt = feature->get_right_undistorted_coord();
            
            // Check if rectified coordinates are valid
            if (right_pt.x < 0 || right_pt.y < 0) {
                // Skip if rectified right coordinate is invalid
                continue;
            }
            
            // Convert to Eigen vectors
            Eigen::Vector2d pix1(left_pt.x, left_pt.y);
            Eigen::Vector2d pix2(right_pt.x, right_pt.y);
            
            // SVD triangulation using rectified projection matrices
            Eigen::Matrix4d A;
            A.row(0) = pix1[0] * P1_eigen.row(2) - P1_eigen.row(0);
            A.row(1) = pix1[1] * P1_eigen.row(2) - P1_eigen.row(1);
            A.row(2) = pix2[0] * P2_eigen.row(2) - P2_eigen.row(0);
            A.row(3) = pix2[1] * P2_eigen.row(2) - P2_eigen.row(1);
            
            Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
            Eigen::Vector4d X_h = svd.matrixV().col(3);
            
            // Check homogeneous coordinate
            if (std::abs(X_h[3]) < 1e-6) {
                homogeneous_fail_count++;
                continue;
            }
            
            // Convert to 3D point
            X_h /= X_h[3];
            Eigen::Vector3d pos_3d = X_h.head<3>();
            
            depth_values.push_back(pos_3d[2]);
            
            // Check depth range
            if (pos_3d[2] < config.getMinDepth() || pos_3d[2] > config.getMaxDepth()) {
                depth_rejected++;
                continue;
            }
            
            // Reprojection error check using rectified projection matrices
            // Project back to left rectified camera
            Eigen::Vector3d proj_left = P1_eigen * X_h;
            Eigen::Vector2d reproj_left(proj_left[0] / proj_left[2], proj_left[1] / proj_left[2]);
            
            // Project back to right rectified camera  
            Eigen::Vector3d proj_right = P2_eigen * X_h;
            Eigen::Vector2d reproj_right(proj_right[0] / proj_right[2], proj_right[1] / proj_right[2]);
            
            // Calculate reprojection errors
            double left_error = (pix1 - reproj_left).norm();
            double right_error = (pix2 - reproj_right).norm();
            double max_error = std::max(left_error, right_error);
            
            if (max_error > config.getMaxReprojectionError()) {
                reprojection_rejected++;
                continue;
            }
            
            // Success! Store the 3D point (transform back to original left camera frame if needed)
            Eigen::Vector3f point3D_float = pos_3d.cast<float>();
            feature->set_3d_point(point3D_float);
            triangulated_count++;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    if (config.isTimingEnabled()) {
        std::cout << "[TIMING] Rectified stereo triangulation: " << duration.count() / 1000.0 << " ms | "
                  << "Triangulated " << triangulated_count << " 3D points from " << total_stereo_matches 
                  << " stereo matches (depth rejected: " << depth_rejected 
                  << ", reprojection rejected: " << reprojection_rejected 
                  << ", homogeneous failed: " << homogeneous_fail_count << ")" << std::endl;
    }
}

} // namespace lightweight_vio
