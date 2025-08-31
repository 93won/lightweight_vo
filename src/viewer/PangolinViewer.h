#pragma once

#include <pangolin/pangolin.h>
#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/scene/axis.h>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>

// Forward declarations
namespace lightweight_vio {
class Feature;
class MapPoint;
}

namespace lightweight_vio {

class PangolinViewer {
public:
    PangolinViewer();
    ~PangolinViewer();

    // Initialization and shutdown
    bool initialize(int width = 1280, int height = 960);
    void shutdown();
    
    // Main loop
    bool should_close() const;
    bool is_ready() const;
    void render();
    
    // Camera control
    void reset_camera();
    
    // Data updates
    void update_points(const std::vector<Eigen::Vector3f>& points);
    void update_pose(const Eigen::Matrix4f& pose);
    void update_trajectory(const std::vector<Eigen::Vector3f>& trajectory);
    void add_ground_truth_pose(const Eigen::Matrix4f& gt_pose);
    void update_ground_truth_trajectory(const std::vector<Eigen::Vector3f>& gt_trajectory);
    
    // Map point updates with color differentiation
    void update_map_points(const std::vector<Eigen::Vector3f>& all_points, const std::vector<Eigen::Vector3f>& current_points);
    
    // Image updates
    void update_tracking_image(const cv::Mat& image);
    void update_tracking_image_with_map_points(const cv::Mat& image, 
                                              const std::vector<std::shared_ptr<Feature>>& features,
                                              const std::vector<std::shared_ptr<MapPoint>>& map_points);
    void update_stereo_image(const cv::Mat& image);
    
    // Input processing
    void process_keyboard_input(bool& auto_play, bool& step_mode, bool& advance_frame);
    void sync_ui_state(bool& auto_play, bool& step_mode);  // Sync UI checkbox with mode state
    void set_space_pressed(bool pressed) { m_space_pressed = pressed; }
    void set_next_pressed(bool pressed) { m_next_pressed = pressed; }
    
    // Button controls for step-by-step processing
    bool is_feed_data_pressed() const { return m_feed_data_pressed; }
    void reset_feed_data_pressed() { m_feed_data_pressed = false; }
    
    // Auto mode control
    bool is_auto_mode_enabled() const;
    
    // Frame info updates
    void update_frame_info(int frame_id, int total_features, int tracked_features, int new_features);
    void update_tracking_stats(int frame_id, int total_features, int stereo_matches, int map_points, float success_rate, float position_error);

private:
    // Pangolin components
    pangolin::OpenGlRenderState s_cam;
    pangolin::View d_cam;
    pangolin::View d_panel;
    pangolin::View d_img_left;
    pangolin::View d_img_right;
    
    // Data storage
    std::vector<Eigen::Vector3f> m_points;
    std::vector<Eigen::Vector3f> m_trajectory;
    std::vector<Eigen::Vector3f> m_gt_trajectory;
    Eigen::Matrix4f m_current_pose;
    
    // Map point storage for color differentiation
    std::vector<Eigen::Vector3f> m_all_map_points;      // White - accumulated map points
    std::vector<Eigen::Vector3f> m_current_map_points;  // Red - current frame tracking points
    
    // Image data
    pangolin::GlTexture m_tracking_image;
    pangolin::GlTexture m_stereo_image;
    bool m_has_tracking_image;
    bool m_has_stereo_image;
    
    // Control variables (simplified - no UI toggles)
    bool m_show_points;
    bool m_show_trajectory;
    bool m_show_gt_trajectory;
    bool m_show_camera_frustum;
    bool m_show_grid;
    bool m_show_axis;
    bool m_follow_camera;
    float m_point_size;
    float m_trajectory_width;
    
    // Tracking debug information
    pangolin::Var<int> m_successful_matches;
    pangolin::Var<int> m_frame_id;
    pangolin::Var<int> m_map_points;
    pangolin::Var<std::string> m_separator;
    
    // Control buttons
    pangolin::Var<bool> m_auto_mode_checkbox;
    pangolin::Var<bool> m_feed_data_button;
    pangolin::Var<bool> m_show_map_point_indices;
    mutable bool m_feed_data_pressed;
    
    // Input state
    bool m_space_pressed;
    bool m_next_pressed;
    
    // Initialization state
    bool m_initialized;
    
    // Window dimensions
    int m_window_width;
    int m_window_height;
    
    // Layout positions
    float m_tracking_image_bottom;
    float m_stereo_image_bottom;
    
    // Thread safety
    mutable std::mutex m_render_mutex;
    bool m_panels_created;
    
    // Drawing functions
    void draw_grid();
    void draw_axis();
    void draw_points();
    void draw_map_points();  // New function for colored map points
    void draw_trajectory();
    void draw_gt_trajectory();
    void draw_pose();
    void draw_camera_frustum();
    void draw_feature_grid(cv::Mat& image);  // Grid overlay for feature distribution
    
    // Utility functions
    pangolin::GlTexture create_texture_from_cv_mat(const cv::Mat& mat);
    void setup_panels();
};

} // namespace lightweight_vio
