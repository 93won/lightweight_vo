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
    
    std::cout << "[TIMING] Feature extraction: " << duration.count() / 1000.0 << " ms | "
              << "Extracted " << corners.size() << " features" << std::endl;
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
                cv::Point2f right_pt_shifted(right_pt.x + right_offset, right_pt.y);
                
                // Draw right feature (red circle)
                cv::circle(combined_image, right_pt_shifted, 3, cv::Scalar(0, 0, 255), 2);
                
                // Draw matching line (yellow)
                cv::line(combined_image, left_pt, right_pt_shifted, cv::Scalar(0, 255, 255), 1);
                
                // Show disparity value
                float disparity = feature->get_stereo_disparity();
                std::string disp_str = cv::format("%.1f", disparity);
                cv::putText(combined_image, disp_str, 
                           cv::Point(left_pt.x + 5, left_pt.y - 5), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.4, 
                           cv::Scalar(255, 255, 255), 1);
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
                            0, 1e-4); // Lower eigenvalue threshold for better tracking

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
            Config::getInstance().getFundamentalThreshold(), 0.99, inlier_mask
        );
        
        std::cout << "Fundamental matrix estimated from " << cv::countNonZero(inlier_mask) 
                  << "/" << good_left_pts.size() << " initial matches" << std::endl;
    }
    
    // Now apply matches with epipolar constraint
    feature_idx = 0;
    for (auto& feature : m_features) {
        if (feature->is_valid() && feature_idx < status.size()) {
            if (status[feature_idx] && err[feature_idx] < 50.0f) {
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
                    if (error > 5.0) {
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
                }
            }
            feature_idx++;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "[TIMING] Stereo matching: " << duration.count() / 1000.0 << " ms | "
              << "Matched " << matches_found << "/" << left_pts.size() << " features"
              << " (using epipolar constraint for unrectified stereo)" << std::endl;
}

} // namespace lightweight_vio
