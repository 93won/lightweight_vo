#include "PangolinViewer.h"
#include <iostream>
#include <cmath>
#include <cstdio>
#include <spdlog/spdlog.h>
#include <util/Config.h>
#include <database/Feature.h>
#include <database/MapPoint.h>

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
    , m_has_tracking_image(false)
    , m_has_stereo_image(false)
    , m_space_pressed(false)
    , m_next_pressed(false)
    , m_feed_data_pressed(false)
    , m_initialized(false)
    , m_window_width(1280)
    , m_window_height(960)
    , m_tracking_image_bottom(0.35f)
    , m_stereo_image_bottom(0.0f)
    , m_panels_created(false)
    , m_show_points(true)
    , m_show_trajectory(true)
    , m_show_gt_trajectory(true)
    , m_show_camera_frustum(true)
    , m_show_grid(true)
    , m_show_axis(true)
    , m_follow_camera(false)
    , m_point_size(3.0f)
    , m_trajectory_width(2.0f)
    , m_successful_matches("ui.Successful matches", 0, 0, get_max_features_from_config())
    , m_frame_id("ui.Frame ID", 0)
    , m_map_points("ui.Num of Map Points", 0)
    , m_separator("ui.─────────", "")
    , m_auto_mode_checkbox("ui.Auto Mode", true, true)
    , m_feed_data_button("ui.Feed Data", false, false)
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
    std::cout << "PangolinViewer initialized successfully" << std::endl;
    return true;
}

void PangolinViewer::setup_panels() {
    // UI 패널 너비를 윈도우 너비의 1/4로 설정
    int ui_panel_width = m_window_width / 4;
    
    // Config에서 실제 이미지 크기 읽어오기
    float image_width, image_height;
    try {
        const auto& config = Config::getInstance();
        image_width = static_cast<float>(config.m_image_width);
        image_height = static_cast<float>(config.m_image_height);
    } catch (const std::exception& e) {
        // 기본값으로 fallback (EuRoC 표준)
        image_width = 752.0f;
        image_height = 480.0f;
    }

    // 2. Feature tracking 이미지 - UI 패널 너비에 맞춰 실제 이미지 비율로 동적 계산
    float tracking_aspect = image_width / image_height;  // 752/480 = 1.567
    float display_width = static_cast<float>(m_window_width) * 0.25f;  // UI 패널 너비 (창 너비의 25%)
    float tracking_height = display_width / tracking_aspect;
    float tracking_normalized_height = tracking_height / static_cast<float>(m_window_height);
    
    // 3. Stereo matching 이미지 - 너비 2배 (좌우 이미지 합쳐짐)
    float stereo_aspect = (image_width * 2.0f) / image_height;  // (752*2)/480 = 3.133
    float stereo_height = display_width / stereo_aspect;
    float stereo_normalized_height = stereo_height / static_cast<float>(m_window_height);
    
    // UI 패널 높이를 동적으로 계산 - 이미지들과 겹치지 않게
    float total_image_height = tracking_normalized_height + stereo_normalized_height;
    float available_space_for_ui = 1.0f - total_image_height;
    float ui_panel_height = std::max(0.2f, available_space_for_ui); // 최소 20% 높이 보장
    
    // 이미지 위치 계산 - 아래쪽부터 차곡차곡
    m_stereo_image_bottom = 0.0f; // 맨 아래
    float stereo_image_top = m_stereo_image_bottom + stereo_normalized_height;
    
    m_tracking_image_bottom = stereo_image_top; // 스테레오 이미지 바로 위
    float tracking_image_top = m_tracking_image_bottom + tracking_normalized_height;
    
    // UI 패널은 이미지들 위쪽 공간에
    float ui_panel_bottom = tracking_image_top;
    float ui_panel_top = 1.0f;
    
    if (!m_panels_created) {
        // 처음에만 패널들을 생성
        
        // UI 패널 너비를 비율로 계산 (윈도우 너비의 1/4)
        float ui_panel_ratio = 0.25f;  // 1/4
        
        // Create main 3D view (takes up most of the screen on the right side)
        d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Frac(ui_panel_ratio), pangolin::Attach::Frac(1.0f))
            .SetHandler(new pangolin::Handler3D(s_cam));

        // 1. UI 패널 - 왼쪽 상단 (이미지들 위쪽)
        d_panel = pangolin::CreatePanel("ui")
            .SetBounds(ui_panel_bottom, ui_panel_top, 0.0, pangolin::Attach::Frac(ui_panel_ratio));
        
        // 2. Feature tracking 이미지
        d_img_left = pangolin::CreateDisplay()
            .SetBounds(m_tracking_image_bottom, tracking_image_top, 0.0, pangolin::Attach::Frac(ui_panel_ratio), -tracking_aspect);
        
        // 3. Stereo matching 이미지
        d_img_right = pangolin::CreateDisplay()
            .SetBounds(m_stereo_image_bottom, stereo_image_top, 0.0, pangolin::Attach::Frac(ui_panel_ratio), -stereo_aspect);
        
        m_panels_created = true;
    } else {
        // 이미 생성된 패널들의 크기만 업데이트 (비율 기반으로)
        float ui_panel_ratio = 0.25f;  // 1/4
        
        // 이미지 크기도 다시 계산 (창 크기 변경 반영)
        float new_display_width = static_cast<float>(m_window_width) * ui_panel_ratio;
        float new_tracking_height = new_display_width / tracking_aspect;
        float new_tracking_normalized_height = new_tracking_height / static_cast<float>(m_window_height);
        
        float new_stereo_height = new_display_width / stereo_aspect;
        float new_stereo_normalized_height = new_stereo_height / static_cast<float>(m_window_height);
        
        // UI 패널 높이를 동적으로 재계산
        float new_total_image_height = new_tracking_normalized_height + new_stereo_normalized_height;
        float new_available_space_for_ui = 1.0f - new_total_image_height;
        float new_ui_panel_height = std::max(0.2f, new_available_space_for_ui);
        
        // 이미지 위치 재계산 - 아래쪽부터 차곡차곡
        m_stereo_image_bottom = 0.0f; // 맨 아래
        float new_stereo_image_top = m_stereo_image_bottom + new_stereo_normalized_height;
        
        m_tracking_image_bottom = new_stereo_image_top; // 스테레오 이미지 바로 위
        float new_tracking_image_top = m_tracking_image_bottom + new_tracking_normalized_height;
        
        // UI 패널은 이미지들 위쪽 공간에
        float new_ui_panel_bottom = new_tracking_image_top;
        float new_ui_panel_top = 1.0f;
        
        d_cam.SetBounds(0.0, 1.0, pangolin::Attach::Frac(ui_panel_ratio), pangolin::Attach::Frac(1.0f));
        d_panel.SetBounds(new_ui_panel_bottom, new_ui_panel_top, 0.0, pangolin::Attach::Frac(ui_panel_ratio));
        
        // 원본 이미지 aspect ratio 유지 (이미지 크기는 bounds로, 비율은 고정)
        d_img_left.SetBounds(m_tracking_image_bottom, new_tracking_image_top, 0.0, pangolin::Attach::Frac(ui_panel_ratio), -tracking_aspect);
        d_img_right.SetBounds(m_stereo_image_bottom, new_stereo_image_top, 0.0, pangolin::Attach::Frac(ui_panel_ratio), -stereo_aspect);
        
        std::cout << "[DEBUG] 패널과 이미지 모두 업데이트 완료: UI 비율=" << ui_panel_ratio 
                  << ", 새 display_width=" << new_display_width 
                  << ", tracking_bottom=" << m_tracking_image_bottom 
                  << ", stereo_bottom=" << m_stereo_image_bottom << std::endl;
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

    // Check Auto mode or Feed Data button
    if (m_auto_mode_checkbox || pangolin::Pushed(m_feed_data_button)) {
        m_feed_data_pressed = true;
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
    glPointSize(m_point_size);
    glColor3f(1.0f, 0.0f, 0.0f); // Red points
    
    glBegin(GL_POINTS);
    for (const auto& point : m_points) {
        glVertex3f(point.x(), point.y(), point.z());
    }
    glEnd();
    
    glPointSize(1.0f);
}

void PangolinViewer::draw_map_points() {
    // Draw accumulated map points in white
    if (!m_all_map_points.empty()) {
        glPointSize(m_point_size);
        glColor3f(1.0f, 1.0f, 1.0f); // White for accumulated map points
        
        glBegin(GL_POINTS);
        for (const auto& point : m_all_map_points) {
            glVertex3f(point.x(), point.y(), point.z());
        }
        glEnd();
    }
    
    // Draw current frame tracking points in red (overlay on top)
    if (!m_current_map_points.empty()) {
        glPointSize(m_point_size + 1.0f); // Slightly larger for visibility
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
    // Get camera pose using T_BC from config
    try {
        const auto& config = Config::getInstance();
        cv::Mat T_bc_cv = config.left_T_BC();
        Eigen::Matrix4f T_bc;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                T_bc(i, j) = T_bc_cv.at<double>(i, j);
            }
        }
        
        // T_cb = T_bc^-1 (body to camera)
        Eigen::Matrix4f T_cb = T_bc.inverse();
        
        // T_wc = T_wb * T_cb (world to camera)
        Eigen::Matrix4f T_wc = m_current_pose * T_cb;
        
        // Draw camera frustum using Pangolin
        glPushMatrix();
        glMultMatrixf(T_wc.data());
        
        const float w = 0.15f;
        const float h = w * 0.75f;
        const float z = w;
        
        glLineWidth(2.0f);
        glColor3f(1.0f, 1.0f, 0.0f); // Yellow frustum
        
        glBegin(GL_LINES);
        
        // Frustum lines from origin to corners
        glVertex3f(0, 0, 0); glVertex3f(w, h, z);
        glVertex3f(0, 0, 0); glVertex3f(w, -h, z);
        glVertex3f(0, 0, 0); glVertex3f(-w, -h, z);
        glVertex3f(0, 0, 0); glVertex3f(-w, h, z);
        
        // Frustum rectangle
        glVertex3f(w, h, z); glVertex3f(w, -h, z);
        glVertex3f(-w, h, z); glVertex3f(-w, -h, z);
        glVertex3f(-w, h, z); glVertex3f(w, h, z);
        glVertex3f(-w, -h, z); glVertex3f(w, -h, z);
        
        glEnd();
        glLineWidth(1.0f);
        
        glPopMatrix();
        
    } catch (const std::exception& e) {
        // If config not available, draw simple frustum
        glPushMatrix();
        glMultMatrixf(m_current_pose.data());
        
        const float w = 0.15f;
        const float h = w * 0.75f;
        const float z = w;
        
        glLineWidth(2.0f);
        glColor3f(1.0f, 1.0f, 0.0f);
        
        glBegin(GL_LINES);
        glVertex3f(0, 0, 0); glVertex3f(w, h, z);
        glVertex3f(0, 0, 0); glVertex3f(w, -h, z);
        glVertex3f(0, 0, 0); glVertex3f(-w, -h, z);
        glVertex3f(0, 0, 0); glVertex3f(-w, h, z);
        glEnd();
        glLineWidth(1.0f);
        
        glPopMatrix();
    }
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

void PangolinViewer::update_trajectory(const std::vector<Eigen::Vector3f>& trajectory) {
    m_trajectory = trajectory;
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
    
    // Just pass through the original image without drawing additional features
    // The image should already have features drawn by Frame::draw_features()
    m_tracking_image = create_texture_from_cv_mat(image);
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
        } else {
            auto_play = true;
            step_mode = false;
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
}

void PangolinViewer::update_frame_info(int frame_id, int total_features, int tracked_features, int new_features) {
    // Frame info is no longer stored in UI variables - just ignore the call
    // This function is kept for compatibility but does nothing
}

void PangolinViewer::update_tracking_stats(int frame_id, int total_features, int stereo_matches, int map_points, float success_rate, float position_error) {
    m_frame_id = frame_id;
    m_successful_matches = stereo_matches;  // Show successful stereo matches
    m_map_points = map_points;
}

bool PangolinViewer::is_auto_mode_enabled() const {
    return m_auto_mode_checkbox;
}

} // namespace lightweight_vio
