#include "PangolinViewer.h"
#include <iostream>
#include <cmath>
#include <cstdio>
#include <spdlog/spdlog.h>
#include <util/Config.h>
#include <database/Feature.h>
#include <database/MapPoint.h>
#include <database/Frame.h>

namespace lightweight_vio {

// Helper function to get max features from config
static int get_max_features_from_config() {
    try {
        const auto& config = Config::getInstance();
        return config.m_max_features;
    } catch (const std::exception& e) {
        // Fallback to default value
        return 150;
    }
}

PangolinViewer::PangolinViewer()
    : m_current_pose(Eigen::Matrix4f::Identity())
    , m_current_camera_pose(Eigen::Matrix4f::Identity())
    , m_relative_pose_from_last_keyframe(Eigen::Matrix4f::Identity())
    , m_has_tracking_image(false)
    , m_has_stereo_image(false)
    , m_space_pressed(false)
    , m_next_pressed(false)
    , m_step_forward_pressed(false)
    , m_initialized(false)
    , m_window_width(1280)
    , m_window_height(960)
    , m_tracking_image_bottom(0.35f)
    , m_stereo_image_bottom(0.0f)
    , m_panels_created(false)
    , m_show_points(true)
    , m_show_trajectory(true)
    , m_show_keyframe_frustums(true)
    , m_show_gt_trajectory(true)
    , m_show_camera_frustum(true)
    , m_show_grid(true)
    , m_show_axis(true)
    , m_follow_camera(false)
    , m_point_size(3.0f)
    , m_trajectory_width(2.0f)
    , m_frame_id("ui.Frame ID", 0)
    , m_successful_matches("ui.Num Tracked Map Points", 0, 0, get_max_features_from_config())
    , m_separator("ui.================================================================================", "")
    , m_auto_mode_checkbox("ui.1. Auto Mode", false, true)
    , m_show_map_point_indices("ui.2. Show Map Point IDs", false, true)
    , m_show_accumulated_map_points("ui.3. Show Accumulated Map Points", true, true)
    , m_step_forward_button("ui.4. Step Forward", false, false)
{
}

PangolinViewer::~PangolinViewer() {
    shutdown();
}

bool PangolinViewer::initialize(int width, int height) {
    // Store window dimensions
    m_window_width = width;
    m_window_height = height;
    
    // Create OpenGL window with Pangolin
    pangolin::CreateWindowAndBind("Lightweight VIO - Pangolin Viewer", width, height);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Set Pangolin UI text color to white for dark theme
    pangolin::RegisterKeyPressCallback(pangolin::PANGO_SPECIAL + pangolin::PANGO_KEY_F1, [](){});
    
    // Setup camera for 3D navigation with proper initial view
    // Use more reasonable focal length based on image dimensions
    float fx = width * 0.7f;  // Adjust focal length to be proportional to window size
    float fy = height * 0.7f;
    s_cam = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(width, height, fx, fy, width/2, height/2, 0.1, 1000),
        pangolin::ModelViewLookAt(-3, -3, 3, 0, 0, 0, pangolin::AxisZ)  // Changed to AxisZ for better orientation
    );

    // Setup display panels
    setup_panels();

    // Set clear color to dark navy background
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);  // Dark navy background

    m_initialized = true;
    return true;
}

void PangolinViewer::setup_panels() {
    // Set UI panel width to 1/4 of window width
    int ui_panel_width = m_window_width / 4;
    
    // Get actual image size from config
    float image_width, image_height;
    try {
        const auto& config = Config::getInstance();
        image_width = static_cast<float>(config.m_image_width);
        image_height = static_cast<float>(config.m_image_height);
    } catch (const std::exception& e) {
        // Fallback to default values (EuRoC standard)
        image_width = 752.0f;
        image_height = 480.0f;
    }

    // 2. Feature tracking image - dynamically calculated based on UI panel width and actual image ratio
    float tracking_aspect = image_width / image_height;  // 752/480 = 1.567
    float display_width = static_cast<float>(m_window_width) * 0.25f;  // UI panel width (25% of window width)
    float tracking_height = display_width / tracking_aspect;
    float tracking_normalized_height = tracking_height / static_cast<float>(m_window_height);
    
    // 3. Stereo matching image - width doubled (left and right images combined)
    float stereo_aspect = (image_width * 2.0f) / image_height;  // (752*2)/480 = 3.133
    float stereo_height = display_width / stereo_aspect;
    float stereo_normalized_height = stereo_height / static_cast<float>(m_window_height);
    
    // Dynamically calculate UI panel height - avoid overlapping with images
    float total_image_height = tracking_normalized_height + stereo_normalized_height;
    float available_space_for_ui = 1.0f - total_image_height;
    float ui_panel_height = std::max(0.2f, available_space_for_ui); // Ensure minimum 20% height
    
    // Calculate image positions - stacking from bottom
    m_stereo_image_bottom = 0.0f; // Bottom-most
    float stereo_image_top = m_stereo_image_bottom + stereo_normalized_height;
    
    m_tracking_image_bottom = stereo_image_top; // Right above stereo image
    float tracking_image_top = m_tracking_image_bottom + tracking_normalized_height;
    
    // UI panel in the space above images
    float ui_panel_bottom = tracking_image_top;
    float ui_panel_top = 1.0f;
    
    if (!m_panels_created) {
        // Create panels only once
        
        // Calculate UI panel width as ratio (1/4 of window width)
        float ui_panel_ratio = 0.25f;  // 1/4
        
        // Create main 3D view (takes up most of the screen on the right side)
        d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Frac(ui_panel_ratio), pangolin::Attach::Frac(1.0f))
            .SetHandler(new pangolin::Handler3D(s_cam));

        // 1. UI panel - top left (above images)
        d_panel = pangolin::CreatePanel("ui")
            .SetBounds(ui_panel_bottom, ui_panel_top, 0.0, pangolin::Attach::Frac(ui_panel_ratio));
        
        // 2. Feature tracking image
        d_img_left = pangolin::CreateDisplay()
            .SetBounds(m_tracking_image_bottom, tracking_image_top, 0.0, pangolin::Attach::Frac(ui_panel_ratio), -tracking_aspect);
        
        // 3. Stereo matching image
        d_img_right = pangolin::CreateDisplay()
            .SetBounds(m_stereo_image_bottom, stereo_image_top, 0.0, pangolin::Attach::Frac(ui_panel_ratio), -stereo_aspect);
        
        m_panels_created = true;
    } else {
        // Update size of already created panels (based on ratio)
        float ui_panel_ratio = 0.25f;  // 1/4
        
        // Recalculate image sizes (reflecting window size changes)
        float new_display_width = static_cast<float>(m_window_width) * ui_panel_ratio;
        float new_tracking_height = new_display_width / tracking_aspect;
        float new_tracking_normalized_height = new_tracking_height / static_cast<float>(m_window_height);
        
        float new_stereo_height = new_display_width / stereo_aspect;
        float new_stereo_normalized_height = new_stereo_height / static_cast<float>(m_window_height);
        
        // Dynamically recalculate UI panel height
        float new_total_image_height = new_tracking_normalized_height + new_stereo_normalized_height;
        float new_available_space_for_ui = 1.0f - new_total_image_height;
        float new_ui_panel_height = std::max(0.2f, new_available_space_for_ui);
        
        // Recalculate image positions - stacking from bottom
        m_stereo_image_bottom = 0.0f; // Bottom-most
        float new_stereo_image_top = m_stereo_image_bottom + new_stereo_normalized_height;
        
        m_tracking_image_bottom = new_stereo_image_top; // Right above stereo image
        float new_tracking_image_top = m_tracking_image_bottom + new_tracking_normalized_height;
        
        // UI panel in the space above images
        float new_ui_panel_bottom = new_tracking_image_top;
        float new_ui_panel_top = 1.0f;
        
        d_cam.SetBounds(0.0, 1.0, pangolin::Attach::Frac(ui_panel_ratio), pangolin::Attach::Frac(1.0f));
        d_panel.SetBounds(new_ui_panel_bottom, new_ui_panel_top, 0.0, pangolin::Attach::Frac(ui_panel_ratio));
        
        // 원본 이미지 aspect ratio 유지 (이미지 크기는 bounds로, 비율은 고정)
        d_img_left.SetBounds(m_tracking_image_bottom, new_tracking_image_top, 0.0, pangolin::Attach::Frac(ui_panel_ratio), -tracking_aspect);
        d_img_right.SetBounds(m_stereo_image_bottom, new_stereo_image_top, 0.0, pangolin::Attach::Frac(ui_panel_ratio), -stereo_aspect);
     
    }
}

void PangolinViewer::shutdown() {
    if (m_initialized) {
        pangolin::DestroyWindow("Lightweight VIO - Pangolin Viewer");
        m_initialized = false;
    }
}

bool PangolinViewer::should_close() const {
    return pangolin::ShouldQuit();
}

bool PangolinViewer::is_ready() const {
    return m_initialized;
}

void PangolinViewer::render() {
    if (!m_initialized) return;
    
    // 매 프레임마다 현재 창 크기를 확인하고 변경사항이 있으면 즉시 적용
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    int current_width = viewport[2];
    int current_height = viewport[3];
    
    // 창 크기가 조금이라도 변경되면 즉시 레이아웃 업데이트
    if (current_width != m_window_width || current_height != m_window_height) {
                  
        m_window_width = current_width;
        m_window_height = current_height;
        
        // 즉시 모든 패널 레이아웃 재설정
        setup_panels();
        
    }

    // Clear screen and activate view to render into
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Activate 3D view
    d_cam.Activate(s_cam);

    // Draw 3D content
    if (m_show_grid) {
        draw_grid();
    }

    if (m_show_axis) {
        draw_axis();
    }

    if (m_show_points && !m_points.empty()) {
        draw_points();
    }

    // Draw map points with color differentiation
    draw_map_points();

    if (m_show_trajectory && !m_trajectory.empty()) {
        draw_trajectory();
    }

    if (m_show_keyframe_frustums && !m_keyframe_window.empty()) {
        draw_keyframe_frustums();
    }

    if (m_show_gt_trajectory && !m_gt_trajectory.empty()) {
        draw_gt_trajectory();
    }

    if (!m_current_pose.isZero()) {
        draw_pose();
        
        if (m_show_camera_frustum) {
            draw_camera_frustum();
        }
    }

    // Render images
    if (m_has_tracking_image) {
        d_img_left.Activate();
        glColor3f(1.0, 1.0, 1.0);
        m_tracking_image.RenderToViewport();
    }

    if (m_has_stereo_image) {
        d_img_right.Activate();
        glColor3f(1.0, 1.0, 1.0);
        m_stereo_image.RenderToViewport();
    }

    // Pangolin automatically renders the UI panel with tracking variables
    // No custom drawing needed - the pangolin::Var variables are displayed automatically

    // Check Step Forward button
    if (pangolin::Pushed(m_step_forward_button)) {
        m_step_forward_pressed = true;
    }

    // Process keyboard input - will be handled externally
    // Note: Space bar and 'n' key handling is done in the main application loop

    // Follow camera if enabled
    if (m_follow_camera && !m_current_pose.isZero()) {
        Eigen::Vector3f pos = m_current_pose.block<3, 1>(0, 3);
        s_cam.Follow(pangolin::OpenGlMatrix::Translate(pos.x(), pos.y(), pos.z()));
    }

    // Ensure UI text is rendered in white for dark theme
    glColor3f(1.0f, 1.0f, 1.0f);

    // Swap frames and Process Events
    pangolin::FinishFrame();
}

void PangolinViewer::reset_camera() {
    s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(-3, -3, 3, 0, 0, 0, pangolin::AxisZ));
}

void PangolinViewer::draw_grid() {
    const float grid_size = 10.0f;
    const float step = 1.0f;
    
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    
    // Grid lines in light gray for dark background
    glColor3f(0.6f, 0.6f, 0.6f);
    for (float i = -grid_size; i <= grid_size; i += step) {
        // X direction lines
        glVertex3f(i, -grid_size, 0.0f);
        glVertex3f(i, grid_size, 0.0f);
        
        // Y direction lines
        glVertex3f(-grid_size, i, 0.0f);
        glVertex3f(grid_size, i, 0.0f);
    }
    
    // Axis lines in different colors
    glColor3f(1.0f, 0.0f, 0.0f); // X axis in red
    glVertex3f(-grid_size, 0.0f, 0.0f);
    glVertex3f(grid_size, 0.0f, 0.0f);
    
    glColor3f(0.0f, 1.0f, 0.0f); // Y axis in green
    glVertex3f(0.0f, -grid_size, 0.0f);
    glVertex3f(0.0f, grid_size, 0.0f);
    
    glEnd();
}

void PangolinViewer::draw_axis() {
    const float axis_length = 2.0f;
    
    glLineWidth(3.0f);
    glBegin(GL_LINES);
    
    // X axis - Red
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(axis_length, 0.0f, 0.0f);
    
    // Y axis - Green
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, axis_length, 0.0f);
    
    // Z axis - Blue
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, axis_length);
    
    glEnd();
    glLineWidth(1.0f);
}

void PangolinViewer::draw_points() {
    glPointSize(m_point_size*5.0f); // Slightly larger for visibility
    glColor3f(1.0f, 0.0f, 0.0f); // Red points
    
    glBegin(GL_POINTS);
    for (const auto& point : m_points) {
        glVertex3f(point.x(), point.y(), point.z());
    }
    glEnd();
    
    glPointSize(1.0f);
}

void PangolinViewer::draw_map_points() {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    
    // Draw all map points in light gray (background points)
    if (!m_all_map_points_storage.empty() && m_show_accumulated_map_points) {
        glPointSize(m_point_size);
        glColor3f(0.8f, 0.8f, 0.8f); // Light gray for all map points
        
        glBegin(GL_POINTS);
        for (const auto& point : m_all_map_points_storage) {
            if (point && !point->is_bad()) {
                Eigen::Vector3f position = point->get_position();
                glVertex3f(position.x(), position.y(), position.z());
            }
        }
        glEnd();
    }
    
    // Draw sliding window map points in white (more prominent)
    if (!m_window_map_points_storage.empty()) {
        glPointSize(m_point_size * 1.5f); // Slightly larger
        glColor3f(1.0f, 1.0f, 1.0f); // White for window map points
        
        glBegin(GL_POINTS);
        for (const auto& point : m_window_map_points_storage) {
            if (point && !point->is_bad()) {
                Eigen::Vector3f position = point->get_position();
                glVertex3f(position.x(), position.y(), position.z());
            }
        }
        glEnd();
    }
    
    // Draw current frame tracking points in red (highest priority - overlay on top)
    if (!m_current_map_points.empty()) {
        glPointSize(m_point_size * 2.0f); // Largest for visibility
        glColor3f(1.0f, 0.0f, 0.0f); // Red for current frame tracking points
        
        glBegin(GL_POINTS);
        for (const auto& point : m_current_map_points) {
            glVertex3f(point.x(), point.y(), point.z());
        }
        glEnd();
    }
    
    glPointSize(1.0f); // Reset point size
}

void PangolinViewer::draw_trajectory() {
    if (m_trajectory.size() < 2) return;
    
    glLineWidth(m_trajectory_width);
    glColor3f(1.0f, 1.0f, 0.0f); // Yellow trajectory
    
    glBegin(GL_LINE_STRIP);
    for (const auto& pos : m_trajectory) {
        glVertex3f(pos.x(), pos.y(), pos.z());
    }
    glEnd();
    
    glLineWidth(1.0f);
}

void PangolinViewer::draw_keyframe_frustums() {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    
    // Use sliding window keyframes instead of old m_keyframe_poses
    if (m_keyframe_window.empty()) return;
    
    // Set sky blue color for keyframe frustums
    glColor3f(0.5f, 0.8f, 1.0f);
    
    for (const auto& keyframe : m_keyframe_window) {
        if (!keyframe) continue;
        
        // Get Twc from keyframe
        Eigen::Matrix4f Twc = keyframe->get_Twc();
        Eigen::Matrix4f T_wc = Twc;
        
        Eigen::Vector3f position = T_wc.block<3, 1>(0, 3);
        Eigen::Matrix3f rotation = T_wc.block<3, 3>(0, 0);
        
        // Draw smaller keyframe frustum
        float scale = 0.1f;  // Smaller scale for keyframes
        
        // Camera frustum vertices (in camera coordinate)
        std::vector<Eigen::Vector3f> frustum_points = {
            Eigen::Vector3f(0, 0, 0),                    // Camera center
            Eigen::Vector3f(-scale, -scale, scale * 2),  // Bottom-left
            Eigen::Vector3f(scale, -scale, scale * 2),   // Bottom-right  
            Eigen::Vector3f(scale, scale, scale * 2),    // Top-right
            Eigen::Vector3f(-scale, scale, scale * 2)    // Top-left
        };
        
        // Transform to world coordinates
        for (auto& point : frustum_points) {
            point = rotation * point + position;
        }
        
        // Draw frustum edges
        glBegin(GL_LINES);
        // Lines from camera center to corners
        for (int i = 1; i < 5; ++i) {
            glVertex3f(frustum_points[0].x(), frustum_points[0].y(), frustum_points[0].z());
            glVertex3f(frustum_points[i].x(), frustum_points[i].y(), frustum_points[i].z());
        }
        // Rectangle at far plane
        for (int i = 1; i < 5; ++i) {
            int next = (i % 4) + 1;
            glVertex3f(frustum_points[i].x(), frustum_points[i].y(), frustum_points[i].z());
            glVertex3f(frustum_points[next].x(), frustum_points[next].y(), frustum_points[next].z());
        }
        glEnd();
    }
}

void PangolinViewer::draw_gt_trajectory() {
    if (m_gt_trajectory.size() < 2) return;
    
    glLineWidth(m_trajectory_width + 1.0f);
    glColor3f(0.0f, 1.0f, 0.0f); // Green GT trajectory
    
    glBegin(GL_LINE_STRIP);
    for (const auto& pos : m_gt_trajectory) {
        glVertex3f(pos.x(), pos.y(), pos.z());
    }
    glEnd();
    
    glLineWidth(1.0f);
}

void PangolinViewer::draw_pose() {
    Eigen::Vector3f position = m_current_pose.block<3, 1>(0, 3);
    Eigen::Matrix3f rotation = m_current_pose.block<3, 3>(0, 0);
    
    // Draw camera position
    glPointSize(8.0f);
    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_POINTS);
    glVertex3f(position.x(), position.y(), position.z());
    glEnd();
    glPointSize(1.0f);
    
    // Draw body frame axes
    const float axis_length = 0.3f;
    glLineWidth(3.0f);
    glBegin(GL_LINES);
    
    // X-axis (Red)
    glColor3f(1.0f, 0.0f, 0.0f);
    Eigen::Vector3f x_axis = position + rotation.col(0) * axis_length;
    glVertex3f(position.x(), position.y(), position.z());
    glVertex3f(x_axis.x(), x_axis.y(), x_axis.z());
    
    // Y-axis (Green)
    glColor3f(0.0f, 1.0f, 0.0f);
    Eigen::Vector3f y_axis = position + rotation.col(1) * axis_length;
    glVertex3f(position.x(), position.y(), position.z());
    glVertex3f(y_axis.x(), y_axis.y(), y_axis.z());
    
    // Z-axis (Blue)
    glColor3f(0.0f, 0.0f, 1.0f);
    Eigen::Vector3f z_axis = position + rotation.col(2) * axis_length;
    glVertex3f(position.x(), position.y(), position.z());
    glVertex3f(z_axis.x(), z_axis.y(), z_axis.z());
    
    glEnd();
    glLineWidth(1.0f);
}

void PangolinViewer::draw_camera_frustum() {
    // Draw current frame frustum using T_wc (same as keyframes)
    // This ensures consistency with keyframe frustum rendering
    
    if (m_current_camera_pose.isZero()) {
        return; // No camera pose available
    }
    
    // Use the stored T_wc directly (same as keyframes)
    Eigen::Vector3f position = m_current_camera_pose.block<3, 1>(0, 3);
    Eigen::Matrix3f rotation = m_current_camera_pose.block<3, 3>(0, 0);
    
    // Draw current frame frustum - larger than keyframes
    float scale = 0.15f;  // Larger scale for current frame
    
    // Camera frustum vertices (in camera coordinate)
    std::vector<Eigen::Vector3f> frustum_points = {
        Eigen::Vector3f(0, 0, 0),                    // Camera center
        Eigen::Vector3f(-scale, -scale, scale * 2),  // Bottom-left
        Eigen::Vector3f(scale, -scale, scale * 2),   // Bottom-right  
        Eigen::Vector3f(scale, scale, scale * 2),    // Top-right
        Eigen::Vector3f(-scale, scale, scale * 2)    // Top-left
    };
    
    // Transform to world coordinates (same as keyframes)
    for (auto& point : frustum_points) {
        point = rotation * point + position;
    }
    
    glLineWidth(2.0f);
    glColor3f(1.0f, 1.0f, 0.0f); // Yellow frustum
    
    // Draw frustum edges
    glBegin(GL_LINES);
    // Lines from camera center to corners
    for (int i = 1; i < 5; ++i) {
        glVertex3f(frustum_points[0].x(), frustum_points[0].y(), frustum_points[0].z());
        glVertex3f(frustum_points[i].x(), frustum_points[i].y(), frustum_points[i].z());
    }
    // Rectangle at far plane
    for (int i = 1; i < 5; ++i) {
        int next = (i % 4) + 1;
        glVertex3f(frustum_points[i].x(), frustum_points[i].y(), frustum_points[i].z());
        glVertex3f(frustum_points[next].x(), frustum_points[next].y(), frustum_points[next].z());
    }
    glEnd();
    glLineWidth(1.0f);
}

pangolin::GlTexture PangolinViewer::create_texture_from_cv_mat(const cv::Mat& mat) {
    pangolin::GlTexture tex;
    
    // Convert BGR to RGB if needed and flip vertically to match OpenGL coordinate system
    cv::Mat rgb_mat;
    if (mat.channels() == 3) {
        cv::cvtColor(mat, rgb_mat, cv::COLOR_BGR2RGB);
    } else if (mat.channels() == 1) {
        cv::cvtColor(mat, rgb_mat, cv::COLOR_GRAY2RGB);
    } else {
        rgb_mat = mat;
    }
    
    // Flip vertically to match OpenGL coordinate system (OpenCV: top-left origin, OpenGL: bottom-left origin)
    cv::Mat flipped_mat;
    cv::flip(rgb_mat, flipped_mat, 0);
    
    tex.Reinitialise(flipped_mat.cols, flipped_mat.rows, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
    tex.Upload(flipped_mat.ptr(), GL_RGB, GL_UNSIGNED_BYTE);
    
    return tex;
}

// Data update functions
void PangolinViewer::update_points(const std::vector<Eigen::Vector3f>& points) {
    m_points = points;
}

void PangolinViewer::update_pose(const Eigen::Matrix4f& pose) {
    m_current_pose = pose;
}

void PangolinViewer::update_camera_pose(const Eigen::Matrix4f& T_wc) {
    m_current_camera_pose = T_wc;
}

void PangolinViewer::update_trajectory(const std::vector<Eigen::Vector3f>& trajectory) {
    m_trajectory = trajectory;
}

// DEPRECATED: Use update_keyframe_window() instead
void PangolinViewer::update_keyframe_poses(const std::vector<Eigen::Matrix4f>& keyframe_poses) {
    // This function is deprecated but kept for backward compatibility
    // Convert to new sliding window format if needed
    spdlog::warn("update_keyframe_poses() is deprecated. Use update_keyframe_window() instead.");
    m_keyframe_poses = keyframe_poses;
}

void PangolinViewer::add_ground_truth_pose(const Eigen::Matrix4f& gt_pose) {
    Eigen::Vector3f position = gt_pose.block<3, 1>(0, 3);
    m_gt_trajectory.push_back(position);
}

void PangolinViewer::update_ground_truth_trajectory(const std::vector<Eigen::Vector3f>& gt_trajectory) {
    m_gt_trajectory = gt_trajectory;
}

void PangolinViewer::update_map_points(const std::vector<Eigen::Vector3f>& all_points, const std::vector<Eigen::Vector3f>& current_points) {
    m_all_map_points = all_points;
    m_current_map_points = current_points;
    
    // spdlog::debug("[VIEWER] Updated map points: {} total, {} current", all_points.size(), current_points.size());
}

void PangolinViewer::update_tracking_image(const cv::Mat& image) {
    if (image.empty()) {
        spdlog::warn("[PangolinViewer] Received empty tracking image");
        return;
    }
    
    m_tracking_image = create_texture_from_cv_mat(image);
    m_has_tracking_image = true;
    
    // The bounds and aspect ratio are now handled exclusively by setup_panels().
    // This function is only responsible for updating the texture.
    // spdlog::debug("[PangolinViewer] Updated tracking image texture {}x{}", image.cols, image.rows);
}

void PangolinViewer::update_tracking_image_with_map_points(const cv::Mat& image, 
                                                          const std::vector<std::shared_ptr<Feature>>& features,
                                                          const std::vector<std::shared_ptr<MapPoint>>& map_points) {
    if (image.empty()) return;
    
    // Create a copy of the image to draw on
    cv::Mat image_with_grid = image.clone();
    
    // Draw grid overlay for feature distribution visualization
    draw_feature_grid(image_with_grid);
    
    // Draw map point indices if enabled (replaces the original blue text in Frame.cpp)
    if (m_show_map_point_indices && !features.empty() && !map_points.empty()) {
        // Simple approach: assume features and map_points are aligned by index
        size_t min_size = std::min(features.size(), map_points.size());
        
        for (size_t i = 0; i < min_size; ++i) {
            if (!features[i] || !features[i]->is_valid()) continue;
            if (!map_points[i] || map_points[i]->is_bad()) continue;
            
            cv::Point2f pixel_coord = features[i]->get_pixel_coord();
            
            // Draw map point ID (red text) - NO CIRCLE DRAWING
            std::string id_text = std::to_string(map_points[i]->get_id());
            cv::Point2f text_pos(pixel_coord.x + 5, pixel_coord.y - 5);  // Offset text slightly
            
            // Use blue color for text (BGR: 255,0,0) and font size 0.7
            cv::putText(image_with_grid, id_text, text_pos, 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);  // Blue text
        }
    }
    
    // Convert to texture
    m_tracking_image = create_texture_from_cv_mat(image_with_grid);
    m_has_tracking_image = true;

    // The bounds and aspect ratio are now handled exclusively by setup_panels().
    // This function is only responsible for updating the texture.
    // spdlog::debug("[PangolinViewer] Updated tracking image with features texture {}x{}", image.cols, image.rows);
}

void PangolinViewer::update_stereo_image(const cv::Mat& image) {
    if (image.empty()) {
        spdlog::warn("[PangolinViewer] Received empty stereo image");
        return;
    }
    
    m_stereo_image = create_texture_from_cv_mat(image);
    m_has_stereo_image = true;

    // The bounds and aspect ratio are now handled exclusively by setup_panels().
    // This function is only responsible for updating the texture.
    // spdlog::debug("[PangolinViewer] Updated stereo image texture {}x{}", image.cols, image.rows);
}

void PangolinViewer::process_keyboard_input(bool& auto_play, bool& step_mode, bool& advance_frame) {
    // Handle space key press - toggle between auto and step mode
    if (m_space_pressed) {
        if (auto_play) {
            auto_play = false;
            step_mode = true;
            m_auto_mode_checkbox = false;  // Update UI checkbox
        } else {
            auto_play = true;
            step_mode = false;
            m_auto_mode_checkbox = true;   // Update UI checkbox
        }
        m_space_pressed = false;
    }
    
    // Handle next frame key press - advance one frame in step mode
    if (m_next_pressed) {
        if (step_mode) {
            advance_frame = true;
        }
        m_next_pressed = false;
    }
    
    // Handle Step Forward button press - advance one frame in step mode
    if (m_step_forward_pressed) {
        if (step_mode) {
            advance_frame = true;
        }
        m_step_forward_pressed = false;
    }
}

void PangolinViewer::sync_ui_state(bool& auto_play, bool& step_mode) {
    // Check if UI checkbox state has changed and update mode accordingly
    bool ui_auto_mode = m_auto_mode_checkbox;
    
    if (ui_auto_mode && !auto_play) {
        // UI checkbox enabled but currently in step mode - switch to auto
        auto_play = true;
        step_mode = false;
    } else if (!ui_auto_mode && auto_play) {
        // UI checkbox disabled but currently in auto mode - switch to step
        auto_play = false;
        step_mode = true;
    }
}

void PangolinViewer::update_frame_info(int frame_id, int total_features, int tracked_features, int new_features) {
    // Frame info is no longer stored in UI variables - just ignore the call
    // This function is kept for compatibility but does nothing
}

void PangolinViewer::update_tracking_stats(int frame_id, int total_features, int stereo_matches, int map_points, float success_rate, float position_error) {
    m_frame_id = frame_id;
    m_successful_matches = stereo_matches;  // Show successful stereo matches
}

bool PangolinViewer::is_auto_mode_enabled() const {
    return m_auto_mode_checkbox;
}

void PangolinViewer::draw_feature_grid(cv::Mat& image) {
    const auto& config = Config::getInstance();
    const int grid_cols = config.m_grid_cols;  // From config
    const int grid_rows = config.m_grid_rows;  // From config
    const cv::Scalar grid_color(100, 100, 100);  // Gray color for grid lines
    const int thickness = 1;
    
    const float cell_width = (float)image.cols / grid_cols;   // ~37.6 pixels per cell
    const float cell_height = (float)image.rows / grid_rows; // ~48.0 pixels per cell
    
    // Draw vertical grid lines (20 divisions)
    for (int i = 1; i < grid_cols; i++) {
        int x = (int)(i * cell_width);
        cv::line(image, cv::Point(x, 0), cv::Point(x, image.rows), grid_color, thickness);
    }
    
    // Draw horizontal grid lines (10 divisions)
    for (int i = 1; i < grid_rows; i++) {
        int y = (int)(i * cell_height);
        cv::line(image, cv::Point(0, y), cv::Point(image.cols, y), grid_color, thickness);
    }
}

// New sliding window keyframe management functions
void PangolinViewer::add_frame(std::shared_ptr<Frame> frame) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_all_frames.push_back(frame);
}

void PangolinViewer::update_keyframe_window(const std::vector<std::shared_ptr<Frame>>& keyframes) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_keyframe_window = keyframes;
}

void PangolinViewer::set_last_keyframe(std::shared_ptr<Frame> last_keyframe) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_last_keyframe = last_keyframe;
}

void PangolinViewer::update_relative_pose_from_last_keyframe(const Eigen::Matrix4f& relative_pose) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_relative_pose_from_last_keyframe = relative_pose;
}

void PangolinViewer::update_all_map_points(const std::vector<std::shared_ptr<MapPoint>>& map_points) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_all_map_points_storage = map_points;
}

void PangolinViewer::update_window_map_points(const std::vector<std::shared_ptr<MapPoint>>& window_map_points) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_window_map_points_storage = window_map_points;
}

} // namespace lightweight_vio
